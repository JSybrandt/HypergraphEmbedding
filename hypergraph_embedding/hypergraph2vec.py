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
from itertools import combinations, permutations, product
from tqdm import tqdm
from collections import namedtuple

import keras
from keras.models import Model
from keras.layers import Input, Embedding, Multiply, Dense
from keras.layers import Dot, Flatten, Average

log = logging.getLogger()


def GetNodeNeighbors(hypergraph):
  """
    Returns a Csr matrix where if i,j=1 then node i and node j share an edge.
  """
  row = []
  col = []
  val = []
  for node_idx, node in hypergraph.node.items():
    for edge_idx in node.edges:
      for neigh_idx in hypergraph.edge[edge_idx].nodes:
        row.append(node_idx)
        col.append(neigh_idx)
        val.append(1)
  return csr_matrix((val, (row, col)), dtype=np.bool)


def GetEdgeNeighbors(hypergraph):
  """
    Returns a Csr matrix where if i,j=1 then edge i and edge j share a node.
    If edge2nodes is set, then we skip the ToCsrMatrix computation
  """
  row = []
  col = []
  val = []
  for edge_idx, edge in hypergraph.edge.items():
    for node_idx in edge.nodes:
      for neigh_idx in hypergraph.node[node_idx].edges:
        row.append(edge_idx)
        col.append(neigh_idx)
        val.append(1)
  return csr_matrix((val, (row, col)), dtype=np.bool)


def _SameTypeSimilarity(row_i, row_j, matrix):
  "Computes the jaccard of two nodes or two edges"
  set_i = matrix[row_i, :]
  set_j = matrix[row_j, :]
  return jaccard(set_i, set_j)


def _NodeEdgeSim(
    node_idx,
    edge_idx,
    node2edges,
    edge2nodes,
    node2neighbors,
    edge2neighbors):
  "Computes the similarity of a node with an edge baised on both neighborhoods"
  edges_containing_node = node2edges[node_idx, :]
  nodes_in_edge = edge2nodes[edge_idx, :]
  return jaccard(node2neighbors[node_idx, :], nodes_in_edge) \
       * jaccard(edge2neighbors[edge_idx, :], edges_containing_node)


def _sample_column_indices(idx, matrix, samples):
  "Returns at most # samples results"
  cols = list(matrix[idx, :].nonzero()[1])
  if len(cols) > samples:
    cols = sample(cols, samples)
  return [c for c in cols]


_shared_info = {}


def _init_precompute_shared_data(
    _node2edges,
    _edge2nodes,
    _node2neighbors,
    _edge2neighbors,
    _all_node_indices,
    _all_edge_indices):
  _shared_info["node2edges"] = _node2edges
  _shared_info["edge2nodes"] = _edge2nodes
  _shared_info["node2neighbors"] = _node2neighbors
  _shared_info["edge2neighbors"] = _edge2neighbors
  _shared_info["all_node_indices"] = _all_node_indices
  _shared_info["all_edge_indices"] = _all_edge_indices


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


def PrecomputeSimilarities(
    hypergraph,
    num_neighbors,
    num_pos_samples_per,
    num_neg_samples_per):
  log.info("Precomputing similarities")

  log.info("Converting hypergraph to node-major sparse matrix")
  node2edges = ToCsrMatrix(hypergraph)

  log.info("Converting hypergraph to edge-major sparse matrix")
  edge2nodes = ToEdgeCsrMatrix(hypergraph)

  log.info("Getting 2nd order node neighbors")
  node2neighbors = GetNodeNeighbors(hypergraph)
  log.info("Getting 2nd order edge neighbors")
  edge2neighbors = GetEdgeNeighbors(hypergraph)

  log.info("Indexing all node indices")
  all_node_indices = set(i for i in hypergraph.node)

  log.info("Indexing all edge indices")
  all_edge_indices = set(i for i in hypergraph.edge)

  similarity_records = []

  # Note, we are going to store indices + 1 because 0 is a mask value

  log.info("Sampling node-node probabilities")
  for node_idx in tqdm(hypergraph.node):
    neighbors = node2neighbors[node_idx]
    pos_indices = set(neighbors.nonzero()[1])
    neg_indices = all_node_indices - pos_indices
    num_pos_samples = min(num_pos_samples_per, len(pos_indices))
    num_neg_samples = min(num_neg_samples_per, len(neg_indices))
    for neigh_idx in sample(pos_indices, num_pos_samples) \
                   + sample(neg_indices, num_neg_samples):
      similarity_records.append(
          SimilarityRecord(
              left_node_idx=node_idx,
              right_node_idx=neigh_idx,
              node_node_prob=_SameTypeSimilarity(
                  node_idx,
                  neigh_idx,
                  node2edges)))

  log.info("Sampling edge-edge probabilities")
  for edge_idx in tqdm(hypergraph.edge):
    neighbors = edge2neighbors[edge_idx]
    pos_indices = set(neighbors.nonzero()[1])
    neg_indices = all_edge_indices - pos_indices
    num_pos_samples = min(num_pos_samples_per, len(pos_indices))
    num_neg_samples = min(num_neg_samples_per, len(neg_indices))
    for neigh_idx in sample(pos_indices, num_pos_samples) \
                   + sample(neg_indices, num_neg_samples):
      similarity_records.append(
          SimilarityRecord(
              left_edge_idx=edge_idx,
              right_edge_idx=neigh_idx,
              edge_edge_prob=_SameTypeSimilarity(
                  edge_idx,
                  neigh_idx,
                  edge2nodes)))

  log.info("Sampling node-edge probabilities")
  for node_idx in tqdm(hypergraph.node):
    pos_indices = set(node2edges[node_idx].nonzero()[1])
    neg_indices = all_edge_indices - pos_indices
    num_pos_samples = min(num_pos_samples_per, len(pos_indices))
    num_neg_samples = min(num_neg_samples_per, len(neg_indices))
    for edge_idx in sample(pos_indices, num_pos_samples) \
                  + sample(neg_indices, num_neg_samples):
      similarity_records.append(
          SimilarityRecord(
              left_node_idx=node_idx,
              right_edge_idx=edge_idx,
              node_edge_prob=_NodeEdgeSim(
                  node_idx,
                  edge_idx,
                  node2edges,
                  edge2nodes,
                  node2neighbors,
                  edge2neighbors),
              edges_containing_node=_sample_column_indices(
                  node_idx,
                  node2edges,
                  num_neighbors),
              nodes_in_edge=_sample_column_indices(
                  edge_idx,
                  edge2nodes,
                  num_neighbors)))

  log.info("Sampling edge-node probabilities")
  for edge_idx in tqdm(hypergraph.edge):
    pos_indices = set(edge2nodes[edge_idx].nonzero()[1])
    neg_indices = all_node_indices - pos_indices
    num_pos_samples = min(num_pos_samples_per, len(pos_indices))
    num_neg_samples = min(num_neg_samples_per, len(neg_indices))
    for node_idx in sample(pos_indices, num_pos_samples) \
                  + sample(neg_indices, num_neg_samples):
      similarity_records.append(
          SimilarityRecord(
              left_node_idx=node_idx,
              right_edge_idx=edge_idx,
              node_edge_prob=_NodeEdgeSim(
                  node_idx,
                  edge_idx,
                  node2edges,
                  edge2nodes,
                  node2neighbors,
                  edge2neighbors),
              edges_containing_node=_sample_column_indices(
                  node_idx,
                  node2edges,
                  num_neighbors),
              nodes_in_edge=_sample_column_indices(
                  edge_idx,
                  edge2nodes,
                  num_neighbors)))
  return _SimilarityValuesToResults(similarity_records, num_neighbors)


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
          Dot(0)([left_node_vec,
                  right_node_vec]))
  edge_edge_prob = Dense(
      1,
      activation="sigmoid",
      name="edge_edge_prob")(
          Dot(0)([left_edge_vec,
                  right_edge_vec]))

  # Get neighborhood embeddings
  nodes_dot_sigs = [
      Dense(1,
            activation="sigmoid")(
                Dot(0)([Flatten()(node_emb(node)),
                        left_node_vec])) for node in nodes_in_edge
  ]
  edges_dot_sigs = [
      Dense(1,
            activation="sigmoid")(
                Dot(0)([Flatten()(edge_emb(edge)),
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

  model.compile(optimizer="rmsprop", loss="kullback_leibler_divergence")
  return model
