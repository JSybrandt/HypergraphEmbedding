# This module contains the new hypergraph2vec process.
# Its primary functions, PrecomputeSimilarities and GetModel
# Are used in the "embedding" module to actually perform this task

from .hypergraph_util import *

import numpy as np
import scipy as sp
from scipy.spatial.distance import jaccard
from scipy.sparse import csr_matrix

from random import random, sample, choice
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

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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


def _similarity_values_to_model_input(similarity_records, num_neighbors):
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
  if pos_samples > 0:
    pos_targets = set(source2targets[source_idx, :].nonzero()[1])
    pos_samples = min(pos_samples, len(pos_targets))
  else:
    pos_targets = set()

  if neg_samples > 0:
    neg_targets = set(idx2features.nonzero()[0]) - pos_targets
    neg_samples = min(neg_samples, len(neg_targets))
  else:
    neg_targets = set()

  for target_idx in sample(pos_targets, pos_samples) \
                  + sample(neg_targets, neg_samples):
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
              node_node_prob=prob))
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
  if pos_samples > 0:
    pos_targets = set(source2targets[source_idx, :].nonzero()[1])
    pos_samples = min(pos_samples, len(pos_targets))
  else:
    pos_targets = set()
  if neg_samples > 0:
    neg_targets = set(target2sources.nonzero()[0]) - pos_targets
    neg_samples = min(neg_samples, len(neg_targets))
  else:
    neg_targets = set()
  for target_idx in sample(pos_targets, pos_samples) \
                  + sample(neg_targets, neg_samples):
    prob = jaccard(source2targets[source_idx], target2neighbors[target_idx]) \
         * jaccard(target2sources[target_idx], source2neighbors[source_idx])

    # Make simpler variable names for assigment
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
  """
  Precomputes node-node, node-edge, and edge-edge similarities for the
  provided hypergraph. This acts as the "observed probabilties" for
  hypergraph2vec.
  input:
    - hypergraph: a hypergraph proto message
    - num_neighbors: the number of 1st degree neghbors to include in output
    - num_pos_samples_per: the number of samples per node/edge in hypergraph
                           where at least one output >0
    - num_neg_samples_per: the number of samples per node/edge in hypergraph
                           where all output is 0
  output:
    - tuple of ([input_features], [outputs]) to match keras model input
  """
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
              num_neg_samples_per, #neg_samples
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
              num_neg_samples_per, #neg_samples
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
  return similarity_records


################################################################################
# Hypergraph2Vec++ Sampler                                                     #
################################################################################


def _init_get_walks(
    idx2features,
    source2target,
    num_walks,
    max_length,
    tolerance,
    source_is_edge):
  _shared_info.clear()
  assert idx2features.shape[0] == source2target.shape[0]
  # walk matrix should be square
  assert source2target.shape[0] == source2target.shape[1]
  assert num_walks > 0
  assert max_length > 0
  assert tolerance > 0
  _shared_info["idx2features"] = idx2features
  _shared_info["source2target"] = source2target
  _shared_info["num_walks"] = num_walks
  _shared_info["max_length"] = max_length
  _shared_info["tolerance"] = tolerance
  _shared_info["source_is_edge"] = source_is_edge


def _get_walks(
    start_idx,
    idx2features=None,
    source2target=None,
    num_walks=None,
    max_length=None,
    tolerance=None,
    source_is_edge=None):
  if idx2features is None:
    idx2features = _shared_info["idx2features"]
  if source2target is None:
    source2target = _shared_info["source2target"]
  if num_walks is None:
    num_walks = _shared_info["num_walks"]
  if max_length is None:
    max_length = _shared_info["max_length"]
  if tolerance is None:
    tolerance = _shared_info["tolerance"]
  if source_is_edge is None:
    source_is_edge = _shared_info["source_is_edge"]

  results = []
  for _ in range(num_walks):
    current_idx = start_idx
    cumulative_jaccard = 1
    for _ in range(max_length):
      neighbor_idx = choice(source2target[current_idx, :].nonzero()[1])
      cumulative_jaccard *= jaccard(idx2features[current_idx, :],
                                    idx2features[neighbor_idx, :])
      if cumulative_jaccard < tolerance:
        break
      else:
        if source_is_edge:
          results.append(
              SimilarityRecord(
                  left_edge_idx=start_idx,
                  right_edge_idx=neighbor_idx,
                  edge_edge_prob=cumulative_jaccard))
        else:
          results.append(
              SimilarityRecord(
                  left_node_idx=start_idx,
                  right_node_idx=neighbor_idx,
                  node_node_prob=cumulative_jaccard))
      neighbor_idx = current_idx
  return results


def PrecomputeSimilaritiesPlusPlus(
    hypergraph,
    num_neighbors,
    walks_per_node,
    max_length,
    tolerance,
    run_in_parallel=True):
  """
  Precomputes node-node, node-edge, and edge-edge similarities for the
  provided hypergraph. This acts as the "observed probabilties" for
  hypergraph2vec.
  input:
    - hypergraph: a hypergraph proto message
  output:
    - tuple of ([input_features], [outputs]) to match keras model input
  """
  assert num_neighbors > 0
  assert walks_per_node > 0
  assert max_length > 0
  assert tolerance > 0
  num_cores = multiprocessing.cpu_count() if run_in_parallel else 1
  # return value
  similarity_records = []

  log.info("Converting hypergraph to node-major sparse matrix")
  node2edges = ToCsrMatrix(hypergraph)
  log.info("Getting 1st order node neighbors")
  node2node_neighbors = node2edges * node2edges.T

  log.info("Sampling node-node probabilities")
  with Pool(num_cores,
            initializer=_init_get_walks,
            initargs=(
              node2edges, #idx2features
              node2node_neighbors, #source2target
              walks_per_node, #num_walks
              max_length, #max_length
              tolerance, #tolerance
              False #source_is_edge
            )) as pool:
    with tqdm(total=len(hypergraph.node)) as pbar:
      for result in pool.imap(_get_walks, hypergraph.node):
        similarity_records += result
        pbar.update(1)

  log.info("Converting hypergraph to edge-major sparse matrix")
  edge2nodes = ToEdgeCsrMatrix(hypergraph)
  log.info("Getting 1st order edge neighbors")
  edge2edge_neighbors = edge2nodes * edge2nodes.T
  log.info("Sampling edge-edge probabilities")
  with Pool(num_cores,
            initializer=_init_get_walks,
            initargs=(
              edge2nodes, #idx2features
              edge2edge_neighbors, #source2targets
              walks_per_node, #num_walks
              max_length, #max_length
              tolerance, #tolerance
              True #source_is_edge
            )) as pool:
    with tqdm(total=len(hypergraph.edge)) as pbar:
      for result in pool.imap(_get_walks, hypergraph.edge):
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
              walks_per_node, #pos_samples
              0, #neg_samples
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
              walks_per_node, #pos_samples
              0, #neg_samples
              True #source_is_edge
            )) as pool:
    with tqdm(total=len(hypergraph.edge)) as pbar:
      for result in pool.imap(_diff_type_sample, hypergraph.edge):
        similarity_records += result
        pbar.update(1)

  log.info("Converting to input for Keras Model")
  return _similarity_values_to_model_input(similarity_records, num_neighbors)


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

################################################################################
# Hooks for runner                                                             #
################################################################################

def EmbedHypergraph(
    hypergraph,
    dimension,
    num_neighbors=5,
    pos_samples=100,
    neg_samples=0,
    batch_size=256,
    epochs=5,
    debug_summary_path=None):
  similarity_records = PrecomputeSimilarities(hypergraph,
                                              num_neighbors,
                                              pos_samples,
                                              neg_samples)
  if debug_summary_path is not None:
    WriteDebugSummary(debug_summary_path, similarity_records)
  input_features, output_probs = _similarity_values_to_model_input(similarity_records, num_neighbors)
  model = GetModel(hypergraph, dimension, num_neighbors)
  model.fit(input_features, output_probs, batch_size=batch_size, epochs=epochs)

  log.info("Recording Embeddings")

  node_weights = model.get_layer("node_embedding").get_weights()[0]
  edge_weights = model.get_layer("edge_embedding").get_weights()[0]

  embedding = HypergraphEmbedding()
  embedding.dim = dimension
  embedding.method_name = "Hypergraph"

  for node_idx in hypergraph.node:
    embedding.node[node_idx].values.extend(node_weights[node_idx + 1])
  for edge_idx in hypergraph.edge:
    embedding.edge[edge_idx].values.extend(edge_weights[edge_idx + 1])
  return embedding

def WriteDebugSummary(debug_summary_path, sim_records):

  log.info("Writing Debug Summary to %s", debug_summary_path)

  nn_probs = [r.node_node_prob for r in sim_records
                            if r.node_node_prob is not None]
  ee_probs = [r.edge_edge_prob for r in sim_records
                            if r.edge_edge_prob is not None]
  ne_probs = [r.node_edge_prob for r in sim_records
                            if r.node_edge_prob is not None]
  fig, (nn_ax, ee_ax, ne_ax) = plt.subplots(3, 1)
  nn_ax.set_title("Node-Node Probability Distribution")
  nn_ax.hist(nn_probs)
  nn_ax.set_yscale("log")
  ee_ax.set_title("Edge-Edge Probability Distribution")
  ee_ax.hist(ee_probs)
  ee_ax.set_yscale("log")
  ne_ax.set_title("Node-Edge Probability Distribution")
  ne_ax.hist(ne_probs)
  ne_ax.set_yscale("log")
  fig.tight_layout()
  fig.savefig(debug_summary_path)
