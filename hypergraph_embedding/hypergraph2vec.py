# This module contains the new hypergraph2vec process.
# Its primary functions, PrecomputeSimilarities and GetModel
# Are used in the "embedding" module to actually perform this task

from .hypergraph_util import *

import numpy as np
import scipy as sp
from scipy.spatial.distance import jaccard
from scipy.sparse import csr_matrix

from random import random, sample
import logging
import multiprocessing
from multiprocessing import Pool
from itertools import combinations, permutations, product
from tqdm import tqdm
from collections import namedtuple

import keras
from keras.models import Model
from keras.layers import Input, Embedding, Multiply, Dense
from keras.layers import Dot, Flatten, Average

################################################################################
# Helper Data - Includes logger and Similarity Record                          #
################################################################################
log = logging.getLogger()

# Used to coordinate between parallel processes
_shared_info = {}

SimilarityRecord = namedtuple(
    "SimilarityRecord",
    (
        "left_node_idx",
        "left_edge_idx",
        "right_node_idx",
        "right_edge_idx",
        "edges_containing_node",
        "nodes_in_edge",
        "node_node_prob",
        "edge_edge_prob",
        "node_edge_prob"))
# Set all field defaults to none
SimilarityRecord.__new__.__defaults__ = (None,) * len(SimilarityRecord._fields)


def _SimilarityValuesToResults(similarity_records, num_neighbors):
  "Converts a list of similarity_records to a tuple of "
  "(feature lists, output_lists)"

  left_node_idx = []
  right_node_idx = []
  left_edge_idx = []
  right_edge_idx = []
  edges_containing_node = [[] for _ in range(num_neighbors)]
  nodes_in_edge = [[] for _ in range(num_neighbors)]
  node_node_prob = []
  edge_edge_prob = []
  node_edge_prob = []

  for r in similarity_records:

    def IncOrZero(x):
      if x is None:
        return 0
      else:
        return x + 1

    def LenOrZero(x):
      if x is None:
        return 0
      else:
        return len(x)

    def ValOrZero(x):
      if x is None:
        return 0
      else:
        return x

    left_node_idx.append(IncOrZero(r.left_node_idx))
    left_edge_idx.append(IncOrZero(r.left_edge_idx))
    right_node_idx.append(IncOrZero(r.right_node_idx))
    right_edge_idx.append(IncOrZero(r.right_edge_idx))
    for i in range(num_neighbors):
      if i >= LenOrZero(r.edges_containing_node):
        edges_containing_node[i].append(0)
      else:
        edges_containing_node[i].append(r.edges_containing_node[i] + 1)
      if i >= LenOrZero(r.nodes_in_edge):
        nodes_in_edge[i].append(0)
      else:
        nodes_in_edge[i].append(r.nodes_in_edge[i] + 1)
    node_node_prob.append(ValOrZero(r.node_node_prob))
    edge_edge_prob.append(ValOrZero(r.edge_edge_prob))
    node_edge_prob.append(ValOrZero(r.node_edge_prob))

  return ([left_node_idx,
           left_edge_idx,
           right_node_idx,
           right_edge_idx] + edges_containing_node + nodes_in_edge,
          [node_node_prob,
           edge_edge_prob,
           node_edge_prob])


################################################################################
# Same Type Sampler - Used to get probabilities in later stages                #
################################################################################


def _init_same_type_sample(
    idx2features,
    source2targets,
    pos_samples,
    neg_samples,
    source_is_edge):
  _shared_info.clear()
  assert idx2features.shape[0] == source2targets.shape[0]
  assert idx2features.shape[0] == source2targets.shape[1]
  assert pos_samples >= 0
  assert neg_samples >= 0
  _shared_info["idx2features"] = idx2features
  _shared_info["source2targets"] = source2targets
  _shared_info["pos_samples"] = pos_samples
  _shared_info["neg_samples"] = neg_samples
  _shared_info["source_is_edge"] = source_is_edge


def _same_type_sample(
    source_idx,
    idx2features=None,
    source2targets=None,
    pos_samples=None,
    neg_samples=None,
    source_is_edge=None):
  if idx2features is None:
    idx2features = _shared_info["idx2features"]
  if source2targets is None:
    source2targets = _shared_info["source2targets"]
  if pos_samples is None:
    pos_samples = _shared_info["pos_samples"]
  if neg_samples is None:
    neg_samples = _shared_info["neg_samples"]
  if source_is_edge is None:
    source_is_edge = _shared_info["source_is_edge"]

  results = []
  pos_targets = set(source2targets[source_idx, :].nonzero()[1])
  neg_targets = set(range(idx2features.shape[0])) - pos_targets
  pos_samples = min(pos_samples, len(pos_targets))
  neg_samples = min(neg_samples, len(neg_targets))
  for target_idx in sample(pos_targets,
                           pos_samples) + sample(neg_targets,
                                                 neg_samples):
    prob = jaccard(idx2features[source_idx, :], idx2features[target_idx, :])
    if source_is_edge:
      results.append(
          SimilarityRecord(
              left_edge_idx=source_idx,
              right_edge_idx=target_idx,
              edge_edge_prob=prob))
    else:
      results.append(
          SimilarityRecord(
              left_node_idx=source_idx,
              right_node_idx=target_idx,
              node_edge_prob=prob))
  return results


################################################################################
# NodeEdgeSampler - Sampler and similarity measure for node-edge connections   #
################################################################################


def _init_diff_type_sample(
    source2targets,
    target2sources,
    source2neighbors,
    target2neighbors,
    num_neighbors,
    pos_samples,
    neg_samples,
    source_is_edge):
  _shared_info.clear()
  assert source2targets.shape[0] == target2sources.shape[1]
  assert source2targets.shape[1] == target2sources.shape[0]
  assert source2neighbors.shape[0] == source2targets.shape[0]
  assert source2neighbors.shape[0] == source2neighbors.shape[1]
  assert target2neighbors.shape[0] == target2sources.shape[0]
  assert target2neighbors.shape[0] == target2neighbors.shape[1]
  assert num_neighbors >= 0
  assert pos_samples >= 0
  assert neg_samples >= 0
  _shared_info["source2targets"] = source2targets
  _shared_info["target2sources"] = target2sources
  _shared_info["source2neighbors"] = source2neighbors
  _shared_info["target2neighbors"] = target2neighbors
  _shared_info["num_neighbors"] = num_neighbors
  _shared_info["pos_samples"] = pos_samples
  _shared_info["neg_samples"] = neg_samples
  _shared_info["source_is_edge"] = source_is_edge


def _diff_type_sample(
    source_idx,
    source2targets=None,
    target2sources=None,
    source2neighbors=None,
    target2neighbors=None,
    num_neighbors=None,
    pos_samples=None,
    neg_samples=None,
    source_is_edge=None):
  if source2targets is None:
    source2targets = _shared_info["source2targets"]
  if target2sources is None:
    target2sources = _shared_info["target2sources"]
  if source2neighbors is None:
    source2neighbors = _shared_info["source2neighbors"]
  if target2neighbors is None:
    target2neighbors = _shared_info["target2neighbors"]
  if num_neighbors is None:
    num_neighbors = _shared_info["num_neighbors"]
  if pos_samples is None:
    pos_samples = _shared_info["pos_samples"]
  if neg_samples is None:
    neg_samples = _shared_info["neg_samples"]
  if source_is_edge is None:
    source_is_edge = _shared_info["source_is_edge"]
  results = []
  pos_targets = set(source2targets[source_idx, :].nonzero()[1])
  neg_targets = set(range(target2sources.shape[0])) - pos_targets
  pos_samples = min(pos_samples, len(pos_targets))
  neg_samples = min(neg_samples, len(neg_targets))
  for target_idx in sample(pos_targets,
                           pos_samples) + sample(neg_targets,
                                                 neg_samples):
    prob = jaccard(source2targets[source_idx], target2neighbors[target_idx]) \
         * jaccard(target2sources[target_idx], source2neighbors[source_idx])
    if source_is_edge:
      node_idx = target_idx
      edge_idx = source_idx
      node2edges = target2sources
      edge2nodes = source2targets
    else:
      node_idx = source_idx
      edge_idx = target_idx
      node2edges = source2targets
      edge2nodes = target2sources

    edges_containing_node = list(node2edges[node_idx, :].nonzero()[1])
    edges_containing_node = sample(
        edges_containing_node,
        min(num_neighbors,
            len(edges_containing_node)))
    nodes_in_edge = list(edge2nodes[edge_idx, :].nonzero()[1])
    nodes_in_edge = sample(
        nodes_in_edge,
        min(num_neighbors,
            len(nodes_in_edge)))

    results.append(
        SimilarityRecord(
            left_node_idx=node_idx,
            right_edge_idx=edge_idx,
            edges_containing_node=edges_containing_node,
            nodes_in_edge=nodes_in_edge,
            node_edge_prob=prob))
  return results


################################################################################
# PrecomputeSimilarities - Used to get similarity measures for hg2v            #
################################################################################


def PrecomputeSimilarities(
    hypergraph,
    num_neighbors,
    num_pos_samples_per,
    num_neg_samples_per,
    run_in_parallel=True):
  num_cores = multiprocessing.cpu_count() if run_in_parallel else 1
  # return value
  similarity_records = []

  log.info("Converting hypergraph to node-major sparse matrix")
  node2edges = ToCsrMatrix(hypergraph)
  log.info("Getting 1st order node neighbors")
  node2node_neighbors = node2edges * node2edges.T
  log.info("Sampling node-node probabilities")
  with Pool(num_cores,
            initializer=_init_same_type_sample,
            initargs=(
              node2edges, #idx2features
              node2node_neighbors, #source2targets
              num_pos_samples_per, #pos_samples
              num_neg_samples_per, #neg_indices
              False #source_is_edge
            )) as pool:
    with tqdm(total=len(hypergraph.node)) as pbar:
      for result in pool.imap(_same_type_sample, hypergraph.node):
        similarity_records += result
        pbar.update(1)

  log.info("Converting hypergraph to edge-major sparse matrix")
  edge2nodes = ToEdgeCsrMatrix(hypergraph)
  log.info("Getting 1st order edge neighbors")
  edge2edge_neighbors = edge2nodes * edge2nodes.T
  log.info("Sampling edge-edge probabilities")
  with Pool(num_cores,
            initializer=_init_same_type_sample,
            initargs=(
              edge2nodes, #idx2features
              edge2edge_neighbors, #source2targets
              num_pos_samples_per, #pos_samples
              num_neg_samples_per, #neg_indices
              True #source_is_edge
            )) as pool:
    with tqdm(total=len(hypergraph.edge)) as pbar:
      for result in pool.imap(_same_type_sample, hypergraph.edge):
        similarity_records += result
        pbar.update(1)

  log.info("Sampling node-edge probabilities")
  with Pool(num_cores,
            initializer=_init_diff_type_sample,
            initargs=(
              node2edges, #source2targets
              edge2nodes, #target2sources
              node2node_neighbors, #source2neighbors
              edge2edge_neighbors, #target2neighbors
              num_neighbors, #num_neighbors
              num_pos_samples_per, #pos_samples
              num_neg_samples_per, #neg_samples
              False #source_is_edge
            )) as pool:
    with tqdm(total=len(hypergraph.node)) as pbar:
      for result in pool.imap(_diff_type_sample, hypergraph.node):
        similarity_records += result
        pbar.update(1)

  log.info("Sampling edge-node probabilities")
  with Pool(num_cores,
            initializer=_init_diff_type_sample,
            initargs=(
              edge2nodes, #  source2targets
              node2edges, #target2sources
              edge2edge_neighbors, #source2neighbors
              node2node_neighbors, #target2neighbors
              num_neighbors, #num_neighbors
              num_pos_samples_per, #pos_samples
              num_neg_samples_per, #neg_samples
              True #source_is_edge
            )) as pool:
    with tqdm(total=len(hypergraph.edge)) as pbar:
      for result in pool.imap(_diff_type_sample, hypergraph.edge):
        similarity_records += result
        pbar.update(1)

  log.info("Converting to input for Keras Model")
  return _SimilarityValuesToResults(similarity_records, num_neighbors)


################################################################################
# Keras Model                                                                  #
################################################################################


def GetModel(hypergraph, dimension, num_neighbors):
  log.info("Constructing Keras Model")

  max_node_idx = max([i for i in hypergraph.node])
  max_edge_idx = max([i for i in hypergraph.edge])

  left_node_idx = Input((1,), name="left_node_idx", dtype=np.int32)
  left_edge_idx = Input((1,), name="left_edge_idx", dtype=np.int32)
  right_node_idx = Input((1,), name="right_node_idx", dtype=np.int32)
  right_edge_idx = Input((1,), name="right_edge_idx", dtype=np.int32)
  edges_containing_node = [
      Input((1,
            ),
            dtype=np.int32,
            name="edges_containing_node_{}".format(i))
      for i in range(num_neighbors)
  ]
  nodes_in_edge = [
      Input((1,
            ),
            dtype=np.int32,
            name="nodes_in_edge_{}".format(i)) for i in range(num_neighbors)
  ]

  node_emb = Embedding(
      input_dim=max_node_idx + 2,
      output_dim=dimension,
      input_length=1,
      name="node_embedding")
  edge_emb = Embedding(
      input_dim=max_edge_idx + 2,
      output_dim=dimension,
      input_length=1,
      name="edge_embedding")

  # Pad each input, get embeddings
  left_node_vec = Flatten()(node_emb(left_node_idx))
  left_edge_vec = Flatten()(edge_emb(left_edge_idx))
  right_node_vec = Flatten()(node_emb(right_node_idx))
  right_edge_vec = Flatten()(edge_emb(right_edge_idx))

  # calculate expected probabilities
  node_node_prob = Dense(
      1,
      activation="sigmoid",
      name="node_node_prob")(
          Dot(1)([left_node_vec,
                  right_node_vec]))
  edge_edge_prob = Dense(
      1,
      activation="sigmoid",
      name="edge_edge_prob")(
          Dot(1)([left_edge_vec,
                  right_edge_vec]))

  # Get neighborhood embeddings
  nodes_dot_sigs = [
      Dense(1,
            activation="sigmoid")(
                Dot(1)([Flatten()(node_emb(node)),
                        left_node_vec])) for node in nodes_in_edge
  ]
  edges_dot_sigs = [
      Dense(1,
            activation="sigmoid")(
                Dot(1)([Flatten()(edge_emb(edge)),
                        right_edge_vec])) for edge in edges_containing_node
  ]

  node_sig_avg = Average()(nodes_dot_sigs)
  edge_sig_avg = Average()(edges_dot_sigs)
  node_edge_prob = Multiply(name="node_edge_prob")([node_sig_avg, edge_sig_avg])
  model = Model(
      inputs=[left_node_idx,
              left_edge_idx,
              right_node_idx,
              right_edge_idx] + edges_containing_node + nodes_in_edge,
      outputs=[node_node_prob,
               edge_edge_prob,
               node_edge_prob])

  model.compile(optimizer="adagrad", loss="kullback_leibler_divergence")
  return model
