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
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

log = logging.getLogger()

_shared_info = {}

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
# WeightedJaccard Samples - helper and sampler                                 #
################################################################################

# Note that the unweighted case is simply wherein node2features, edge2features = 1


def SparseWeightedJaccard(row_i, row_j):
  "row_i and row_j are sparse vectors of numbers"
  assert row_i.shape[0] == 1
  assert row_j.shape[0] == 1
  assert row_i.shape[1] == row_j.shape[1]

  nz_i = row_i.nonzero()[1]
  nz_j = row_j.nonzero()[1]

  important = np.union1d(nz_i, nz_j)

  numerator = denominator = 0
  for col_idx in important:
    x = row_i[0, col_idx]
    y = row_j[0, col_idx]
    if x < y:
      numerator += x
      denominator += y
    else:
      numerator += y
      denominator += x

  if denominator == 0:
    return 0
  return numerator / denominator


def SparseVecToWeights(vec, idx2weight):
  "vec is a sparse boolean vector, idx2weight has a weight for each col"
  assert vec.shape[0] == 1
  tmp = lil_matrix(vec.shape, dtype=np.float32)
  for col in vec.nonzero()[1]:
    tmp[0, col] = idx2weight[col]
  return csr_matrix(tmp)


## Parallel Helper Functions ###################################################


def _alpha_scale(val, alpha=0):
  assert alpha >= 0
  assert alpha <= 1
  assert val <= 1
  assert val >= 0
  return alpha + (1 - alpha) * (val)


def _init_same_type_sample(
    idx2neighbors,
    num_samples,
    idx2target,
    idx2features,
    is_edge):
  _shared_info.clear()
  _shared_info["idx2neighbors"] = idx2neighbors
  _shared_info["num_samples"] = num_samples
  _shared_info["idx2target"] = idx2target
  _shared_info["idx2features"] = idx2features
  _shared_info["is_edge"] = is_edge


def SameTypeSample(
    idx,
    idx2neighbors=None,
    num_samples=None,
    idx2target=None,
    idx2features=None,
    is_edge=None):
  if idx2neighbors is None:
    idx2neighbors = _shared_info["idx2neighbors"]
  if num_samples is None:
    num_samples = _shared_info["num_samples"]
  if idx2target is None:
    idx2target = _shared_info["idx2target"]
  if idx2features is None:
    idx2features = _shared_info["idx2features"]
  if is_edge is None:
    is_edge = _shared_info["is_edge"]
  records = []
  neighbors = idx2neighbors[idx].nonzero()[1]
  vec = idx2features[idx]
  for neighbor_idx in np.random.choice(neighbors,
                                       min(num_samples, len(neighbors)),
                                       replace=False):
    prob = SparseWeightedJaccard(vec, idx2features[neighbor_idx])
    if is_edge:
      records.append(
          SimilarityRecord(
              left_edge_idx=idx,
              right_edge_idx=neighbor_idx,
              edge_edge_prob=_alpha_scale(prob)))
    else:
      records.append(
          SimilarityRecord(
              left_node_idx=idx,
              right_node_idx=neighbor_idx,
              node_node_prob=_alpha_scale(prob)))
  return records


## Different Type Sample #######################################################


def _init_node_edge_sample(
    node2edge,
    edge2node,
    node2second_edge,
    edge2second_node,
    num_neighbors,
    num_samples,
    node2features,
    edge2features):
  _shared_info.clear()
  _shared_info["node2edge"] = node2edge
  _shared_info["edge2node"] = edge2node
  _shared_info["node2second_edge"] = node2second_edge
  _shared_info["edge2second_node"] = edge2second_node
  _shared_info["num_neighbors"] = num_neighbors
  _shared_info["num_samples"] = num_samples
  _shared_info["node2features"] = node2features
  _shared_info["edge2features"] = edge2features


def _node_edge_sample(node_idx, edge_idx):
  "Calculates exactly one sample. REQUIRES INIT"
  node2edge = _shared_info["node2edge"]
  edge2node = _shared_info["edge2node"]
  num_neighbors = _shared_info["num_neighbors"]
  node2features = _shared_info["node2features"]
  edge2features = _shared_info["edge2features"]

  edges = list(node2edge[node_idx].nonzero()[1])
  # Sample from edges
  neighbor_edge_indices = sample(edges, min(num_neighbors, len(edges)))
  nodes = list(edge2node[edge_idx, :].nonzero()[1])
  # Sample from nodes
  neighbor_node_indices = sample(nodes, min(num_neighbors, len(nodes)))

  node_centroid = (
      edge2node[edge_idx] * node2features) / edge2node[edge_idx].nnz
  prob_by_node = SparseWeightedJaccard(node2features[node_idx], node_centroid)

  edge_centroid = (
      node2edge[node_idx] * edge2features) / node2edge[node_idx].nnz
  prob_by_edge = SparseWeightedJaccard(edge2features[edge_idx], edge_centroid)

  return SimilarityRecord(
      left_node_idx=node_idx,
      right_edge_idx=edge_idx,
      left_weight=node2features[node_idx,
                                edge_idx],
      right_weight=edge2features[edge_idx,
                                 node_idx],
      neighbor_node_indices=neighbor_node_indices,
      neighbor_edge_indices=neighbor_edge_indices,
      node_edge_prob=_alpha_scale(prob_by_node * prob_by_edge))


def SampleNodeEdgePerNode(node_idx):
  node2second_edge = _shared_info["node2second_edge"]
  num_samples = _shared_info["num_samples"]

  records = []
  edges = list(node2second_edge[node_idx].nonzero()[1])
  for edge_idx in sample(edges, min(num_samples, len(edges))):
    records.append(_node_edge_sample(node_idx, edge_idx))
  return records


def SampleNodeEdgePerEdge(edge_idx):
  edge2second_node = _shared_info["edge2second_node"]
  num_samples = _shared_info["num_samples"]

  records = []
  nodes = list(edge2second_node[edge_idx].nonzero()[1])
  for node_idx in sample(nodes, min(num_samples, len(nodes))):
    records.append(_node_edge_sample(node_idx, edge_idx))
  return records


def WeightedJaccardSamples(
    hypergraph,
    node2features,
    edge2features,
    num_neighbors,
    num_samples,
    run_in_parallel=True,
    disable_pbar=False):
  """
  This function performs samples wherein the node-node, node-edge, and
  edge-edge jaccard calculations are based on node2features, and edge2features
  respectively. This is intended to allow for size or span-based weighting.
  Input:
    hypergraph: Hypergraph proto message
    node2features: a matrix where each row corresponds with a feature vector
    edge2features: a matrix where each row corresponds with a feature vector
  """

  log.info("Performing input checks")
  assert num_neighbors >= 0
  assert num_samples >= 0

  workers = multiprocessing.cpu_count() if run_in_parallel else 1

  # return value
  similarity_records = []

  log.info("Converting hypergraph to node-major sparse matrix")
  node2edge = ToCsrMatrix(hypergraph)
  log.info("Checking that feature matrix agrees with node-major sparse matrix")
  assert node2edge.shape[0] == node2features.shape[0]
  log.info("Getting 1st order node neighbors")
  node2node_neighbors = node2edge * node2edge.T

  log.info("Sampling node-node probabilities")
  with Pool(workers,
            initializer=_init_same_type_sample,
            initargs=(node2node_neighbors,  # idx2neighbors
                      num_samples,  # num_samples
                      node2edge,  # idx2target
                      node2features, # idx2features
                      False  # is_edge
                     )) as pool:
    for records in tqdm(pool.imap(SameTypeSample,
                                  hypergraph.node),
                        total=len(hypergraph.node),
                        disable=disable_pbar):
      similarity_records.extend(records)

  log.info("Converting hypergraph to edge-major sparse matrix")
  edge2node = ToEdgeCsrMatrix(hypergraph)
  log.info("Checking that feature matrix agrees with edge-major sparse matrix")
  assert edge2node.shape[0] == edge2features.shape[0]
  log.info("Getting 1st order edge neighbors")
  edge2edge_neighbors = edge2node * edge2node.T

  log.info("Sampling edge-edge probabilities")
  with Pool(workers,
            initializer=_init_same_type_sample,
            initargs=(edge2edge_neighbors,  # idx2neighbors
                      num_samples,  # num_samples
                      edge2node,  # idx2target
                      edge2features, # idx2features
                      True  # is_edge
                     )) as pool:
    for records in tqdm(pool.imap(SameTypeSample,
                                  hypergraph.edge),
                        total=len(hypergraph.edge),
                        disable=disable_pbar):
      similarity_records.extend(records)

  node2second_edge = node2node_neighbors * node2edge
  edge2second_node = edge2edge_neighbors * edge2node

  with Pool(workers,
            initializer=_init_node_edge_sample,
            initargs=(node2edge,
                      edge2node,
                      node2second_edge,
                      edge2second_node,
                      num_neighbors,
                      num_samples,
                      node2features,
                      edge2features)) as pool:
    log.info("Getting node-edge relationships")
    for records in tqdm(pool.imap(SampleNodeEdgePerNode,
                                  hypergraph.node),
                        total=len(hypergraph.node),
                        disable=disable_pbar):
      similarity_records.extend(records)
    log.info("Getting edge-node relationships")
    for records in tqdm(pool.imap(SampleNodeEdgePerEdge,
                                  hypergraph.edge),
                        total=len(hypergraph.edge),
                        disable=disable_pbar):
      similarity_records.extend(records)

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


################################################################################
# Visualization - Plot out the distribution of weights and samples             #
################################################################################


def PlotDistributions(debug_summary_path, sim_records):

  log.info("Writing Debug Summary to %s", debug_summary_path)
  node2features = {}
  edge2features = {}
  for r in sim_records:
    if r.left_weight is not None:
      if r.left_node_idx is not None and r.left_node_idx not in node2features:
        node2features[r.left_node_idx] = r.left_weight
      if r.left_edge_idx is not None and r.left_edge_idx not in edge2features:
        edge2features[r.left_edge_idx] = r.left_weight
    if r.right_weight is not None:
      if r.right_node_idx is not None and r.right_node_idx not in node2features:
        node2features[r.right_node_idx] = r.right_weight
      if r.right_edge_idx is not None and r.right_edge_idx not in edge2features:
        edge2features[r.right_edge_idx] = r.right_weight

  nn_probs = [
      r.node_node_prob for r in sim_records if r.node_node_prob is not None
  ]
  ee_probs = [
      r.edge_edge_prob for r in sim_records if r.edge_edge_prob is not None
  ]
  ne_probs = [
      r.node_edge_prob for r in sim_records if r.node_edge_prob is not None
  ]
  fig, (node_spans,
        edge_spans,
        nn_ax,
        ee_ax,
        ne_ax) = plt.subplots(5, 1, figsize=(8.5, 11))
  node_spans.set_title("Node Weights")
  node_spans.hist(list(node2features.values()))
  node_spans.set_yscale("log")
  edge_spans.set_title("Edge Weights")
  edge_spans.hist(list(edge2features.values()))
  edge_spans.set_yscale("log")
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

  log.info("Finished Writing")
