################################################################################
# This module is responsible for housing all  of the custom sampling functions #
# Each function takes a hypergraph and optional parameters, and reports a list #
# of Similarity Records. These records can be converted into model input later #
################################################################################

from .hypergraph_util import *

from collections import namedtuple
import logging
import multiprocessing
from multiprocessing import Pool
from random import sample
from tqdm import tqdm

log = logging.getLogger()

SimilarityRecord = namedtuple(
    "SimilarityRecord",
    ("left_node_idx",
     "left_edge_idx",
     "right_node_idx",
     "right_edge_idx",
     "left_weight",  # Only used in weighted cases
     "right_weight",  # Only used in weighted cases
     "neighbor_node_indices",
     "neighbor_node_weights",  # Only used in weighted cases
     "neighbor_edge_indices",
     "neighbor_edge_weights",  # Only used in weighted cases
     "node_node_prob",
     "edge_edge_prob",
     "node_edge_prob"))
# Set all field defaults to none
SimilarityRecord.__new__.__defaults__ = (None,) * len(SimilarityRecord._fields)


def BooleanSamples(hypergraph, num_neighbors=5, num_samples=250):
  """
  This function samples num_samples times (at most) for each node-node,
  node-edge, edge-node, and edge-edge connection.
  If two nodes share an edge, two edges share a node, or if a node occurs
  in an edge, then their reported probability is 1. Otherwise 0.
  This does not produce any negative samples.
  """

  # return value
  similarity_records = []

  log.info("Converting hypergraph to node-major sparse matrix")
  node2edge = ToCsrMatrix(hypergraph)
  log.info("Getting 1st order node neighbors")
  node2node_neighbors = node2edge * node2edge.T

  log.info("Sampling node-node probabilities")
  for row_idx in tqdm(hypergraph.node):
    cols = list(node2node_neighbors[row_idx, :].nonzero()[1])
    for col_idx in sample(cols, min(num_neighbors, len(cols))):
      similarity_records.append(
          SimilarityRecord(
              left_node_idx=row_idx,
              right_node_idx=col_idx,
              node_node_prob=1))

  log.info("Converting hypergraph to edge-major sparse matrix")
  edge2node = ToEdgeCsrMatrix(hypergraph)
  log.info("Getting 1st order edge neighbors")
  edge2edge_neighbors = edge2node * edge2node.T

  log.info("Sampling edge-edge probabilities")
  for row_idx in tqdm(hypergraph.edge):
    cols = list(edge2edge_neighbors[row_idx, :].nonzero()[1])
    for col_idx in sample(cols, min(num_neighbors, len(cols))):
      similarity_records.append(
          SimilarityRecord(
              left_edge_idx=row_idx,
              right_edge_idx=col_idx,
              edge_edge_prob=1))

  log.info("Getting node-edge relationships")
  for node_idx in tqdm(hypergraph.node):
    edges = list(node2edge[node_idx, :].nonzero()[1])
    for edge_idx in sample(edges, min(num_neighbors, len(edges))):
      neighbor_edge_indices = list(node2edge[node_idx, :].nonzero()[1])
      neighbor_edge_indices = sample(
          neighbor_edge_indices,
          min(num_neighbors,
              len(neighbor_edge_indices)))
      neighbor_node_indices = list(edge2node[edge_idx, :].nonzero()[1])
      neighbor_node_indices = sample(
          neighbor_node_indices,
          min(num_neighbors,
              len(neighbor_node_indices)))
      similarity_records.append(
          SimilarityRecord(
              left_node_idx=node_idx,
              right_edge_idx=edge_idx,
              neighbor_node_indices=neighbor_node_indices,
              neighbor_edge_indices=neighbor_edge_indices,
              node_edge_prob=1))

  log.info("Getting edge-node relationships")
  for edge_idx in tqdm(hypergraph.edge):
    nodes = list(edge2node[edge_idx, :].nonzero()[1])
    for node_idx in sample(nodes, min(num_neighbors, len(nodes))):
      neighbor_edge_indices = list(node2edge[node_idx, :].nonzero()[1])
      neighbor_edge_indices = sample(
          neighbor_edge_indices,
          min(num_neighbors,
              len(neighbor_edge_indices)))
      neighbor_node_indices = list(edge2node[edge_idx, :].nonzero()[1])
      neighbor_node_indices = sample(
          neighbor_node_indices,
          min(num_neighbors,
              len(neighbor_node_indices)))
      similarity_records.append(
          SimilarityRecord(
              left_node_idx=node_idx,
              right_edge_idx=edge_idx,
              neighbor_node_indices=neighbor_node_indices,
              neighbor_edge_indices=neighbor_edge_indices,
              node_edge_prob=1))

  return similarity_records


################################################################################
# AdjJaccardSamples - Helper functions and sampler                             #
################################################################################


def SparseBooleanJaccard(sparse_a, sparse_b):
  set_a = set(sparse_a.nonzero()[1])
  set_b = set(sparse_b.nonzero()[1])
  return len(set_a.intersection(set_b)) / len(set_a.union(set_b))


def AdjJaccardSamples(hypergraph, num_neighbors=5, num_samples=250):
  """
  This function samples num_samples times (at most) for each node-node,
  node-edge, edge-node, and edge-edge connection.
  If two nodes share an edges, then we estimate their connectivity to be the
  jaccard of their edge sets. If two edges share a node, their connectivity
  is the jaccard of their node sets. If a node is present in an edge, their
  connectivity is the jaccard of the node's 1st order neighbors with the
  edge's 2nd order neighbors, times the jaccard of the edge's 1st order
  neighbors times the node's 2nd order neighbors.
  """

  # return value
  similarity_records = []

  log.info("Converting hypergraph to node-major sparse matrix")
  node2edge = ToCsrMatrix(hypergraph)
  log.info("Getting 1st order node neighbors")
  node2node_neighbors = node2edge * node2edge.T

  log.info("Sampling node-node probabilities")
  for row_idx in tqdm(hypergraph.node):
    cols = list(node2node_neighbors[row_idx, :].nonzero()[1])
    for col_idx in sample(cols, min(num_neighbors, len(cols))):
      prob = SparseBooleanJaccard(node2edge[row_idx], node2edge[col_idx])
      similarity_records.append(
          SimilarityRecord(
              left_node_idx=row_idx,
              right_node_idx=col_idx,
              node_node_prob=prob))

  log.info("Converting hypergraph to edge-major sparse matrix")
  edge2node = ToEdgeCsrMatrix(hypergraph)
  log.info("Getting 1st order edge neighbors")
  edge2edge_neighbors = edge2node * edge2node.T

  log.info("Sampling edge-edge probabilities")
  for row_idx in tqdm(hypergraph.edge):
    cols = list(edge2edge_neighbors[row_idx, :].nonzero()[1])
    for col_idx in sample(cols, min(num_neighbors, len(cols))):
      prob = SparseBooleanJaccard(edge2node[row_idx], edge2node[col_idx])
      similarity_records.append(
          SimilarityRecord(
              left_edge_idx=row_idx,
              right_edge_idx=col_idx,
              edge_edge_prob=prob))

  log.info("Getting node-edge relationships")
  for node_idx in tqdm(hypergraph.node):
    edges = list(node2edge[node_idx, :].nonzero()[1])
    for edge_idx in sample(edges, min(num_neighbors, len(edges))):
      neighbor_edge_indices = list(node2edge[node_idx, :].nonzero()[1])
      neighbor_edge_indices = sample(
          neighbor_edge_indices,
          min(num_neighbors,
              len(neighbor_edge_indices)))
      neighbor_node_indices = list(edge2node[edge_idx, :].nonzero()[1])
      neighbor_node_indices = sample(
          neighbor_node_indices,
          min(num_neighbors,
              len(neighbor_node_indices)))
      prob_by_node = SparseBooleanJaccard(
          edge2node[edge_idx],
          node2node_neighbors[node_idx])
      prob_by_edge = SparseBooleanJaccard(
          node2edge[node_idx],
          edge2edge_neighbors[edge_idx])
      similarity_records.append(
          SimilarityRecord(
              left_node_idx=node_idx,
              right_edge_idx=edge_idx,
              neighbor_node_indices=neighbor_node_indices,
              neighbor_edge_indices=neighbor_edge_indices,
              node_edge_prob=prob_by_node * prob_by_edge))

  log.info("Getting edge-node relationships")
  for edge_idx in tqdm(hypergraph.edge):
    nodes = list(edge2node[edge_idx, :].nonzero()[1])
    for node_idx in sample(nodes, min(num_neighbors, len(nodes))):
      neighbor_edge_indices = list(node2edge[node_idx, :].nonzero()[1])
      neighbor_edge_indices = sample(
          neighbor_edge_indices,
          min(num_neighbors,
              len(neighbor_edge_indices)))
      neighbor_node_indices = list(edge2node[edge_idx, :].nonzero()[1])
      neighbor_node_indices = sample(
          neighbor_node_indices,
          min(num_neighbors,
              len(neighbor_node_indices)))
      prob_by_node = SparseBooleanJaccard(
          edge2node[edge_idx],
          node2node_neighbors[node_idx])
      prob_by_edge = SparseBooleanJaccard(
          node2edge[node_idx],
          edge2edge_neighbors[edge_idx])
      similarity_records.append(
          SimilarityRecord(
              left_node_idx=node_idx,
              right_edge_idx=edge_idx,
              neighbor_node_indices=neighbor_node_indices,
              neighbor_edge_indices=neighbor_edge_indices,
              node_edge_prob=prob_by_node * prob_by_edge))

  return similarity_records


################################################################################
# Samples to Model Input w/ Helper functions                                   #
################################################################################


def _inc_or_zero(x):
  if x is None:
    return 0
  else:
    return x + 1


def _val_or_zero(x):
  if x is None:
    return 0
  else:
    return x


def _pad_or_val(arrOrNone, idx):
  if arrOrNone is None or idx >= len(arrOrNone):
    return 0
  return arrOrNone[idx]


def _pad_or_inc(arrOrNone, idx):
  if arrOrNone is None or idx >= len(arrOrNone):
    return 0
  return arrOrNone[idx] + 1


def SamplesToModelInput(similarity_records, num_neighbors, weighted=True):
  "Converts the above named tuple into (input arrays, output arrays)"
  left_node_idx = []
  left_edge_idx = []
  right_node_idx = []
  right_edge_idx = []
  left_weight = []
  right_weight = []
  neighbor_node_indices = [[] for _ in range(num_neighbors)]
  neighbor_node_weights = [[] for _ in range(num_neighbors)]
  neighbor_edge_indices = [[] for _ in range(num_neighbors)]
  neighbor_edge_weights = [[] for _ in range(num_neighbors)]
  node_node_prob = []
  edge_edge_prob = []
  node_edge_prob = []

  for r in similarity_records:
    left_node_idx.append(_inc_or_zero(r.left_node_idx))
    right_node_idx.append(_inc_or_zero(r.right_node_idx))
    left_edge_idx.append(_inc_or_zero(r.left_edge_idx))
    right_edge_idx.append(_inc_or_zero(r.right_edge_idx))
    left_weight.append(_val_or_zero(
        r.left_weight))  # if not supplied, set to bad
    right_weight.append(_val_or_zero(r.right_weight))
    for i in range(num_neighbors):
      neighbor_node_indices[i].append(_pad_or_inc(r.neighbor_node_indices, i))
      neighbor_edge_indices[i].append(_pad_or_inc(r.neighbor_edge_indices, i))
      if weighted:
        neighbor_node_weights[i].append(_pad_or_val(r.neighbor_node_weights, i))
        neighbor_edge_weights[i].append(_pad_or_val(r.neighbor_edge_weights, i))
    node_node_prob.append(_val_or_zero(r.node_node_prob))
    edge_edge_prob.append(_val_or_zero(r.edge_edge_prob))
    node_edge_prob.append(_val_or_zero(r.node_edge_prob))

  features = [left_node_idx, left_edge_idx, right_node_idx, right_edge_idx]
  if weighted:
    features += [left_weight, right_weight]
  features += neighbor_node_indices
  if weighted:
    features += neighbor_node_weights
  features += neighbor_edge_indices
  if weighted:
    features += neighbor_edge_weights

  targets = [node_node_prob, edge_edge_prob, node_edge_prob]

  return (features, targets)
