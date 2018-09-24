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
from scipy.sparse import coo_matrix

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


def GetSamples(matrix, interesting_rows, samples_per_row):
  res = []
  for row_idx in tqdm(interesting_rows):
    cols = matrix[row_idx, :].nonzero()[1]
    samples = min(samples_per_row, len(cols))
    for col_idx in np.random.choice(cols, samples, replace=False):
      res.append((row_idx, col_idx))
  return res


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


def _init_centroid_from_rows(idx2targets, targets2features):
  _shared_info.clear()
  _shared_info["idx2targets"] = idx2targets
  _shared_info["targets2features"] = targets2features


def CentroidFromRows(idx, idx2targets=None, targets2features=None):
  if idx2targets is None:
    idx2targets = _shared_info["idx2targets"]
  if targets2features is None:
    targets2features = _shared_info["targets2features"]

  rows = idx2targets[idx].nonzero()[1]
  centroid = targets2features[rows].sum(axis=0) / len(rows)
  rows = []
  cols = []
  vals = []
  for col_idx in centroid.nonzero()[1]:
    rows.append(idx)
    cols.append(col_idx)
    vals.append(centroid[0, col_idx])
  return (vals, (rows, cols))


def GetAllCentroids(important_indices, idx2targets, targets2features):
  rows = []
  cols = []
  vals = []
  with Pool(multiprocessing.cpu_count(),
            initializer=_init_centroid_from_rows,
            initargs=(idx2targets,
                      targets2features)) as pool:
    for v, (r, c) in tqdm(pool.imap(CentroidFromRows,
                                    important_indices,
                                    chunksize=4),
                          total=len(important_indices)):
      vals.extend(v)
      rows.extend(r)
      cols.extend(c)

  return csr_matrix((vals, (rows, cols)))


## Parallel Helper Functions ###################################################


def _alpha_scale(val, alpha=0):
  assert alpha >= 0
  assert alpha <= 1
  assert val <= 1
  assert val >= 0
  return alpha + (1 - alpha) * (val)


def _init_same_type_sample(idx2features, is_edge):
  _shared_info.clear()
  _shared_info["idx2features"] = idx2features
  _shared_info["is_edge"] = is_edge


def SameTypeSample(indices, idx2features=None, is_edge=None):
  if idx2features is None:
    idx2features = _shared_info["idx2features"]
  if is_edge is None:
    is_edge = _shared_info["is_edge"]

  idx, neighbor_idx = indices
  prob = SparseWeightedJaccard(idx2features[idx], idx2features[neighbor_idx])
  if is_edge:
    return SimilarityRecord(
        left_edge_idx=idx,
        right_edge_idx=neighbor_idx,
        edge_edge_prob=_alpha_scale(prob))
  else:
    return SimilarityRecord(
        left_node_idx=idx,
        right_node_idx=neighbor_idx,
        node_node_prob=_alpha_scale(prob))


## Different Type Sample #######################################################


def _init_node_edge_sample(
    node2edge,
    edge2node,
    num_neighbors,
    node2features,
    edge2features,
    node2edge_centroid,
    edge2node_centroid):
  _shared_info.clear()
  _shared_info["node2edge"] = node2edge
  _shared_info["edge2node"] = edge2node
  _shared_info["num_neighbors"] = num_neighbors
  _shared_info["node2features"] = node2features
  _shared_info["edge2features"] = edge2features
  _shared_info["node2edge_centroid"] = node2edge_centroid
  _shared_info["edge2node_centroid"] = edge2node_centroid


def EdgeNodeSample(indices):
  edge_idx, node_idx = indices
  return NodeEdgeSample((node_idx, edge_idx))


def NodeEdgeSample(
    indices,
    node2edge=None,
    edge2node=None,
    num_neighbors=None,
    node2features=None,
    edge2features=None,
    node2edge_centroid=None,
    edge2node_centroid=None):

  node_idx, edge_idx = indices
  if node2edge is None:
    node2edge = _shared_info["node2edge"]
  if edge2node is None:
    edge2node = _shared_info["edge2node"]
  if num_neighbors is None:
    num_neighbors = _shared_info["num_neighbors"]
  if node2features is None:
    node2features = _shared_info["node2features"]
  if edge2features is None:
    edge2features = _shared_info["edge2features"]
  if node2edge_centroid is None:
    node2edge_centroid = _shared_info["node2edge_centroid"]
  if edge2node_centroid is None:
    edge2node_centroid = _shared_info["edge2node_centroid"]

  edges = node2edge[node_idx].nonzero()[1]
  neighbor_edge_indices = np.random.choice(
      edges,
      min(num_neighbors,
          len(edges)),
      replace=False)

  nodes = edge2node[edge_idx, :].nonzero()[1]
  neighbor_node_indices = np.random.choice(
      nodes,
      min(num_neighbors,
          len(nodes)),
      replace=False)

  prob_by_node = SparseWeightedJaccard(
      node2features[node_idx],
      edge2node_centroid[edge_idx])
  prob_by_edge = SparseWeightedJaccard(
      edge2features[edge_idx],
      node2edge_centroid[node_idx])

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

  log.info("Getting node-node samples")
  samples = GetSamples(node2node_neighbors, hypergraph.node, num_samples)

  log.info("Sampling node-node probabilities")

  with Pool(workers,
            initializer=_init_same_type_sample,
            initargs=(node2features, # idx2features
                      False  # is_edge
                     )) as pool:
    for record in tqdm(pool.imap(SameTypeSample,
                                 samples,
                                 chunksize=num_samples),
                       total=len(samples),
                       disable=disable_pbar):
      similarity_records.append(record)

  log.info("Converting hypergraph to edge-major sparse matrix")
  edge2node = ToEdgeCsrMatrix(hypergraph)
  log.info("Checking that feature matrix agrees with edge-major sparse matrix")
  assert edge2node.shape[0] == edge2features.shape[0]
  log.info("Getting 1st order edge neighbors")
  edge2edge_neighbors = edge2node * edge2node.T

  log.info("Getting edge-edge samples")
  samples = GetSamples(edge2edge_neighbors, hypergraph.edge, num_samples)

  log.info("Sampling edge-edge probabilities")
  with Pool(workers,
            initializer=_init_same_type_sample,
            initargs=(edge2features, # idx2features
                      True  # is_edge
                     )) as pool:
    for record in tqdm(pool.imap(SameTypeSample,
                                 samples,
                                 chunksize=num_samples),
                       total=len(samples),
                       disable=disable_pbar):
      similarity_records.append(record)

  log.info("Getting node centroids")
  node2edge_centroid = GetAllCentroids(
      hypergraph.node,
      node2edge,
      edge2features)
  log.info("Getting edge centroids")
  edge2node_centroid = GetAllCentroids(
      hypergraph.edge,
      edge2node,
      node2features)

  node2second_edge = node2node_neighbors * node2edge
  log.info("Getting node-edge samples")
  samples = GetSamples(node2second_edge, hypergraph.node, num_samples)
  with Pool(workers,
            initializer=_init_node_edge_sample,
            initargs=(node2edge,
                      edge2node,
                      num_neighbors,
                      node2features,
                      edge2features,
                      node2edge_centroid,
                      edge2node_centroid)) as pool:
    log.info("Getting node-edge relationships")
    for record in tqdm(pool.imap(NodeEdgeSample,
                                 samples),
                       total=len(samples),
                       disable=disable_pbar):
      similarity_records.append(record)

  edge2second_node = edge2edge_neighbors * edge2node
  log.info("Getting edge-node samples")
  samples = GetSamples(edge2second_node, hypergraph.edge, num_samples)
  with Pool(workers,
            initializer=_init_node_edge_sample,
            initargs=(node2edge,
                      edge2node,
                      num_neighbors,
                      node2features,
                      edge2features,
                      node2edge_centroid,
                      edge2node_centroid)) as pool:
    log.info("Getting edge-node relationships")
    for record in tqdm(pool.imap(EdgeNodeSample,
                                 samples),
                       total=len(samples),
                       disable=disable_pbar):
      similarity_records.append(record)

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
