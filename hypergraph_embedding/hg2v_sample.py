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
from random import choice
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
import math

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

log = logging.getLogger()

_shared_info = {}

SimilarityRecord = namedtuple(
    "SimilarityRecord",
    (
        "left_node_idx",
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


def _sample_neighbors(idx, idx2neighbors, num_neighbors):
  return np.random.choice(
      idx2neighbors[idx].nonzero()[1], num_neighbors, replace=True)

def _sample_adj_matrix(matrix,
                       interesting_rows,
                       samples_per_row,
                       disable_pbar=False,
                       replace=False,
                       negative=False):
  """
  Each row represents an object. Its nonzero columns represent connections.
  This function returns a list of (row, col) indices sampled for each row.
  If negative is set, we rely on the matrix having no unused rows
  or cols.
  """
  if type(samples_per_row) == int:
    samples_per_row = [samples_per_row for _ in interesting_rows]
  assert len(samples_per_row) == len(interesting_rows)
  assert len(samples_per_row) > 0
  res = []
  for row_idx, num_samples in tqdm(zip(interesting_rows, samples_per_row),
                                   total=len(interesting_rows),
                                   disable=disable_pbar):
    if negative:
      for col_idx in np.random.randint(matrix.shape[1], size=num_samples):
        res.append((row_idx, col_idx))
    else:
      cols = matrix[row_idx, :].nonzero()[1]
      # a coarse version of the HG could result in an isolated node or edge
      if len(cols) > 0:
        if replace == False:
          samples = min(num_samples, len(cols))
        else:
          samples = num_samples
        for col_idx in np.random.choice(cols, samples, replace=replace):
          res.append((row_idx, col_idx))
  return res


def _alpha_scale(val, alpha=0):
  assert alpha >= 0
  assert alpha <= 1
  assert val <= 1
  assert val >= 0
  return alpha + (1 - alpha) * (val)


## Parallel init functions #####################################################


def _init_same_type_sample(idx2features, is_edge):
  "used for both jaccard and distance sample"
  _shared_info.clear()
  _shared_info["idx2features"] = idx2features
  _shared_info["is_edge"] = is_edge


def _init_diff_type_sample(node2edge, edge2node, num_neighbors, node2features,
                           edge2features, node2edge_centroid,
                           edge2node_centroid):
  _shared_info.clear()
  _shared_info["node2edge"] = node2edge
  _shared_info["edge2node"] = edge2node
  _shared_info["num_neighbors"] = num_neighbors
  _shared_info["node2features"] = node2features
  _shared_info["edge2features"] = edge2features
  _shared_info["node2edge_centroid"] = node2edge_centroid  # not used in dist
  _shared_info["edge2node_centroid"] = edge2node_centroid  # not used in dist


################################################################################
# BooleanSamples                                                               #
################################################################################


def BooleanSamples(hypergraph,
                   num_neighbors,
                   num_samples,
                   neg_samples=0,
                   disable_pbar=False):
  """
  This function samples num_samples times (at most) for each node-node,
  node-edge, edge-node, and edge-edge connection.
  If two nodes share an edge, two edges share a node, or if a node occurs
  in an edge, then their reported probability is 1. Otherwise 0.
  This does not produce any negative samples.
  """

  node_samples = [int(node.weight * num_samples)
                  for _, node in hypergraph.node.items()]
  edge_samples = [int(edge.weight * num_samples)
                  for _, edge in hypergraph.edge.items()]

  neg_node_samples = [int(node.weight * neg_samples)
                  for _, node in hypergraph.node.items()]
  neg_edge_samples = [int(edge.weight * neg_samples)
                  for _, edge in hypergraph.edge.items()]

  # return value
  similarity_records = []

  log.info("Converting hypergraph to node-major sparse matrix")
  node2edge = ToCsrMatrix(hypergraph)
  log.info("Getting 1st order node neighbors")
  node2node_neighbors = node2edge * node2edge.T

  log.info("Sampling node-node probabilities")
  samples = _sample_adj_matrix(node2node_neighbors, hypergraph.node,
                               node_samples, disable_pbar)
  for node_i, node_j in tqdm(samples, disable=disable_pbar):
    similarity_records.append(
        SimilarityRecord(
            left_node_idx=node_i, right_node_idx=node_j, node_node_prob=1))

  log.info("Converting hypergraph to edge-major sparse matrix")
  edge2node = ToEdgeCsrMatrix(hypergraph)
  log.info("Getting 1st order edge neighbors")
  edge2edge_neighbors = edge2node * edge2node.T

  log.info("Sampling edge-edge probabilities")
  samples = _sample_adj_matrix(edge2edge_neighbors, hypergraph.edge,
                               edge_samples, disable_pbar)
  for edge_i, edge_j in tqdm(samples, disable=disable_pbar):
    similarity_records.append(
        SimilarityRecord(
            left_edge_idx=edge_i, right_edge_idx=edge_j, edge_edge_prob=1))

  log.info("Getting node-edge relationships")
  samples = _sample_adj_matrix(node2edge, hypergraph.node, node_samples,
                               disable_pbar)
  log.info("Getting edge-node relationships")
  samples.extend([(n, e) for e, n in _sample_adj_matrix(
      edge2node, hypergraph.edge, edge_samples, disable_pbar)])
  for node_idx, edge_idx in tqdm(samples, disable=disable_pbar):
    neighbor_edge_indices = _sample_neighbors(node_idx, node2edge,
                                              num_neighbors)
    neighbor_node_indices = _sample_neighbors(edge_idx, edge2node,
                                              num_neighbors)
    similarity_records.append(
        SimilarityRecord(
            left_node_idx=node_idx,
            right_edge_idx=edge_idx,
            neighbor_node_indices=neighbor_node_indices,
            neighbor_edge_indices=neighbor_edge_indices,
            node_edge_prob=1))


  # NEGATIVE SAMPLING
  if neg_samples > 0:

    log.info("Node-Node Negatives")
    samples = _sample_adj_matrix(node2node_neighbors, hypergraph.node,
                                 neg_node_samples, disable_pbar,
                                 negative=True)
    for node_i, node_j in tqdm(samples, disable=disable_pbar):
      similarity_records.append(SimilarityRecord(left_node_idx=node_i,
                                                 right_node_idx=node_j))

    log.info("Edge-Edge Negatives")
    samples = _sample_adj_matrix(edge2edge_neighbors, hypergraph.edge,
                                 neg_edge_samples, disable_pbar,
                                 negative=True)
    for node_i, node_j in tqdm(samples, disable=disable_pbar):
      similarity_records.append(SimilarityRecord(left_edge_idx=node_i,
                                                 right_edge_idx=node_j))
    log.info("Node-Edge Negatives")
    samples = _sample_adj_matrix(edge2edge_neighbors, hypergraph.edge,
                                 neg_edge_samples, disable_pbar,
                                 negative=True)
    for node_i, node_j in tqdm(samples, disable=disable_pbar):
      similarity_records.append(SimilarityRecord(left_edge_idx=node_i,
                                                 right_edge_idx=node_j))

    log.info("Getting node-edge negatives")
    samples = _sample_adj_matrix(node2edge, hypergraph.node, neg_node_samples,
                                 disable_pbar, negative=True)
    log.info("Getting edge-node negatives")
    samples.extend([(n, e) for e, n in _sample_adj_matrix(
        edge2node, hypergraph.edge, neg_edge_samples, disable_pbar,
        negative=True)])
    for node_idx, edge_idx in tqdm(samples, disable=disable_pbar):
      neighbor_edge_indices = _sample_neighbors(node_idx, node2edge,
                                                num_neighbors)
      neighbor_node_indices = _sample_neighbors(edge_idx, edge2node,
                                                num_neighbors)
      similarity_records.append(
          SimilarityRecord(
              left_node_idx=node_idx,
              right_edge_idx=edge_idx,
              neighbor_node_indices=neighbor_node_indices,
              neighbor_edge_indices=neighbor_edge_indices))

  return similarity_records


################################################################################
# WeightedJaccard Samples - helper and sampler                                 #
################################################################################


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


def GetAllCentroids(important_indices, idx2targets, targets2features,
                    disable_pbar):
  rows = []
  cols = []
  vals = []
  with Pool(
      multiprocessing.cpu_count(),
      initializer=_init_centroid_from_rows,
      initargs=(idx2targets, targets2features)) as pool:
    for v, (r, c) in tqdm(
        pool.imap(CentroidFromRows, important_indices, chunksize=16),
        total=len(important_indices),
        disable=disable_pbar):
      vals.extend(v)
      rows.extend(r)
      cols.extend(c)

  log.info("Converting centroids to csr_matrix")
  return csr_matrix((vals, (rows, cols)),
                    shape=(idx2targets.shape[0], targets2features.shape[1]))


def SameTypeJaccardSample(indices, idx2features=None, is_edge=None):
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


def DiffTypeJaccardSample(indices,
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

  assert node2edge.shape[0] == node2features.shape[0]
  assert node2edge.shape[0] == node2edge_centroid.shape[0]
  assert node2edge.shape[1] == edge2node.shape[0]
  assert edge2node.shape[0] == edge2features.shape[0]
  assert edge2node.shape[0] == edge2node_centroid.shape[0]

  assert edge2features.shape[1] == node2edge_centroid.shape[1]
  assert node2features.shape[1] == edge2node_centroid.shape[1]

  neighbor_edge_indices = _sample_neighbors(node_idx, node2edge, num_neighbors)
  neighbor_node_indices = _sample_neighbors(edge_idx, edge2node, num_neighbors)

  prob_by_node = SparseWeightedJaccard(node2features[node_idx],
                                       edge2node_centroid[edge_idx])
  prob_by_edge = SparseWeightedJaccard(edge2features[edge_idx],
                                       node2edge_centroid[node_idx])

  return SimilarityRecord(
      left_node_idx=node_idx,
      right_edge_idx=edge_idx,
      left_weight=prob_by_node,
      right_weight=prob_by_edge,
      neighbor_node_indices=neighbor_node_indices,
      neighbor_edge_indices=neighbor_edge_indices,
      node_edge_prob=_alpha_scale(prob_by_node * prob_by_edge))


def WeightedJaccardSamples(hypergraph,
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

  node_samples = [int(node.weight * num_samples)
                  for _, node in hypergraph.node.items()]
  edge_samples = [int(edge.weight * num_samples)
                  for _, edge in hypergraph.edge.items()]

  # return value
  similarity_records = []

  log.info("Converting hypergraph to node-major sparse matrix")
  node2edge = ToCsrMatrix(hypergraph)
  log.info("Checking that feature matrix agrees with node-major sparse matrix")
  assert node2edge.shape[0] == node2features.shape[0]
  log.info("Getting 1st order node neighbors")
  node2node_neighbors = node2edge * node2edge.T

  log.info("Getting node-node samples")
  samples = _sample_adj_matrix(node2node_neighbors, hypergraph.node,
                               node_samples, disable_pbar)
  log.info("Sampling node-node probabilities")
  with Pool(
      workers,
      initializer=_init_same_type_sample,
      initargs=(
          node2features,  # idx2features
          False  # is_edge
      )) as pool:
    for record in tqdm(
        pool.imap(SameTypeJaccardSample, samples, chunksize=num_samples),
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
  samples = _sample_adj_matrix(edge2edge_neighbors, hypergraph.edge,
                               edge_samples, disable_pbar)
  log.info("Sampling edge-edge probabilities")
  with Pool(
      workers,
      initializer=_init_same_type_sample,
      initargs=(
          edge2features,  # idx2features
          True  # is_edge
      )) as pool:
    for record in tqdm(
        pool.imap(SameTypeJaccardSample, samples, chunksize=num_samples),
        total=len(samples),
        disable=disable_pbar):
      similarity_records.append(record)

  log.info("Getting node centroids")
  node2edge_centroid = GetAllCentroids(hypergraph.node, node2edge,
                                       edge2features, disable_pbar)
  log.info("Node centroids have ~%f nonzero entries per row",
           node2edge_centroid.nnz / len(hypergraph.node))

  log.info("Getting edge centroids")
  edge2node_centroid = GetAllCentroids(hypergraph.edge, edge2node,
                                       node2features, disable_pbar)
  log.info("Edge centroids have ~%f nonzero entries per row",
           edge2node_centroid.nnz / len(hypergraph.edge))

  node2second_edge = node2node_neighbors * node2edge
  log.info("Getting node-edge samples")
  samples = _sample_adj_matrix(node2second_edge, hypergraph.node, node_samples,
                               disable_pbar)

  edge2second_node = edge2edge_neighbors * edge2node
  log.info("Getting edge-node samples")
  edge_samples = _sample_adj_matrix(edge2second_node, hypergraph.edge,
                                    edge_samples, disable_pbar)
  edge_samples = [(n, e) for e, n in edge_samples]
  samples.extend(edge_samples)

  with Pool(
      workers,
      initializer=_init_diff_type_sample,
      initargs=(node2edge, edge2node, num_neighbors, node2features,
                edge2features, node2edge_centroid, edge2node_centroid)) as pool:
    log.info("Getting node-edge relationships")
    for record in tqdm(
        pool.imap(DiffTypeJaccardSample, samples),
        total=len(samples),
        disable=disable_pbar):
      similarity_records.append(record)

  return similarity_records


################################################################################
# AlgebraicDistanceSamples - With helpers                                      #
################################################################################


def _init_same_type_distance_sample(idx2target, source_half_emb,
                                    target_half_emb, is_edge):
  _shared_info.clear()
  _shared_info['idx2target'] = idx2target
  _shared_info['source_half_emb'] = source_half_emb
  _shared_info['target_half_emb'] = target_half_emb
  _shared_info['is_edge'] = is_edge


def _same_type_dist_calc(indices, idx2target, source_half_emb, target_half_emb):
  shared_targets = set(idx2target[indices[0]].nonzero()[1]).intersection(
      set(idx2target[indices[1]].nonzero()[1]))
  if len(shared_targets) == 0:
    return 0

  emb_i = np.array(source_half_emb[indices[0]].values, dtype=np.float32)
  emb_j = np.array(source_half_emb[indices[1]].values, dtype=np.float32)

  prob = 0
  max_dist = math.sqrt(len(emb_i))
  for target_idx in shared_targets:
    emb_k = np.array(target_half_emb[target_idx].values, dtype=np.float32)
    weight_ik = (max_dist - np.linalg.norm(emb_i - emb_k, ord=2)) / max_dist
    weight_jk = (max_dist - np.linalg.norm(emb_j - emb_k, ord=2)) / max_dist
    prob = max(prob, min(weight_ik, weight_jk))
  return prob


def SameTypeDistanceSample(indices,
                           idx2target=None,
                           source_half_emb=None,
                           target_half_emb=None,
                           is_edge=None):
  """
  Instead of calculating weighted jaccard, we are going to calculate the
  max/min connection.
  """
  if idx2target is None:
    idx2target = _shared_info["idx2target"]
  if source_half_emb is None:
    source_half_emb = _shared_info["source_half_emb"]
  if target_half_emb is None:
    target_half_emb = _shared_info["target_half_emb"]
  if is_edge is None:
    is_edge = _shared_info["is_edge"]

  prob = _same_type_dist_calc(indices, idx2target, source_half_emb,
                              target_half_emb)

  if is_edge:
    return SimilarityRecord(
        left_edge_idx=indices[0],
        right_edge_idx=indices[1],
        edge_edge_prob=_alpha_scale(prob))
  else:
    return SimilarityRecord(
        left_node_idx=indices[0],
        right_node_idx=indices[1],
        node_node_prob=_alpha_scale(prob))


def _init_diff_type_distance_sample(node2edge, edge2node, num_neighbors,
                                    algebraic_embedding):
  _shared_info.clear()
  _shared_info["node2edge"] = node2edge
  _shared_info["edge2node"] = edge2node
  _shared_info["num_neighbors"] = num_neighbors
  _shared_info["algebraic_embedding"] = algebraic_embedding


def DiffTypeDistanceSample(indices,
                           node2edge=None,
                           edge2node=None,
                           num_neighbors=None,
                           algebraic_embedding=None):

  node_idx, edge_idx = indices
  if node2edge is None:
    node2edge = _shared_info["node2edge"]
  if edge2node is None:
    edge2node = _shared_info["edge2node"]
  if num_neighbors is None:
    num_neighbors = _shared_info["num_neighbors"]
  if algebraic_embedding is None:
    algebraic_embedding = _shared_info["algebraic_embedding"]

  neighbor_edge_indices = _sample_neighbors(node_idx, node2edge, num_neighbors)
  neighbor_node_indices = _sample_neighbors(edge_idx, edge2node, num_neighbors)

  node_prob = 0
  """
  # This couple of lines may lead to a more accurate, but less feasible solution.
  for other_node_idx in edge2node[edge_idx].nonzero()[1]:
    node_prob = max(node_prob,
                  _same_type_dist_calc((node_idx, other_node_idx),
                                       node2edge,
                                       algebraic_embedding.node,
                                       algebraic_embedding.edge))
  """
  edge_prob = 0
  for other_edge_idx in node2edge[node_idx].nonzero()[1]:
    edge_prob = max(
        edge_prob,
        _same_type_dist_calc((edge_idx, other_edge_idx), edge2node,
                             algebraic_embedding.edge,
                             algebraic_embedding.node))
  return SimilarityRecord(
      left_node_idx=node_idx,
      right_edge_idx=edge_idx,
      neighbor_node_indices=neighbor_node_indices,
      neighbor_edge_indices=neighbor_edge_indices,
      node_edge_prob=_alpha_scale(max(node_prob, edge_prob)))


def AlgebraicDistanceSamples(hypergraph,
                             algebraic_embedding,
                             num_neighbors,
                             num_samples,
                             run_in_parallel=True,
                             disable_pbar=False):
  """
    This function samples node-node, edge-edge, and node-edge relationships
    directly accounting for algebraic distance. That is, the probability that
    two nodes share an edge, two edges share a node, or a node exists within
    an edge, is modeled directly by algebraic distance.
  """

  log.info("Performing input checks")
  assert num_neighbors >= 0
  assert num_samples >= 0

  # workers = multiprocessing.cpu_count() if run_in_parallel else 1
  workers = min(8, multiprocessing.cpu_count()) if run_in_parallel else 1

  # return value
  similarity_records = []

  node2edge = ToCsrMatrix(hypergraph)
  edge2node = ToEdgeCsrMatrix(hypergraph)

  log.info("Getting node-node samples")
  samples = _sample_adj_matrix(node2edge * node2edge.T, hypergraph.node,
                               num_samples, disable_pbar)

  log.info("Sampling node-node probabilities")
  with Pool(
      workers,
      initializer=_init_same_type_distance_sample,
      initargs=(
          node2edge,  # idx2target
          algebraic_embedding.node,  # source half
          algebraic_embedding.edge,  # target half
          False  # is_edge
      )) as pool:
    for record in tqdm(
        pool.imap(SameTypeDistanceSample, samples, chunksize=num_samples),
        total=len(samples),
        disable=disable_pbar):
      similarity_records.append(record)

  log.info("Getting edge-edge samples")
  samples = _sample_adj_matrix(edge2node * edge2node.T, hypergraph.edge,
                               num_samples, disable_pbar)
  log.info("Sampling edge-edge probabilities")
  with Pool(
      workers,
      initializer=_init_same_type_distance_sample,
      initargs=(
          edge2node,  # idx2target
          algebraic_embedding.edge,  # source emb
          algebraic_embedding.node,  # target emb
          True  # is_edge
      )) as pool:
    for record in tqdm(
        pool.imap(SameTypeDistanceSample, samples, chunksize=num_samples),
        total=len(samples),
        disable=disable_pbar):
      similarity_records.append(record)

  log.info("Getting node-edge samples")
  samples = _sample_adj_matrix(node2edge * node2edge.T * node2edge,
                               hypergraph.node, num_samples, disable_pbar)
  log.info("Getting edge-node samples")
  samples += [
      (n, e)
      for e, n in _sample_adj_matrix(edge2node * edge2node.T * edge2node,
                                     hypergraph.edge, num_samples, disable_pbar)
  ]
  with Pool(
      workers,
      initializer=_init_diff_type_distance_sample,
      initargs=(node2edge, edge2node, num_neighbors,
                algebraic_embedding)) as pool:
    for record in tqdm(
        pool.imap(DiffTypeDistanceSample, samples),
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
  fig, (node_spans, edge_spans, nn_ax, ee_ax, ne_ax) = plt.subplots(
      5, 1, figsize=(8.5, 11))
  if len(node2features.values()):
    node_spans.set_title("Node Weights")
    node_spans.hist(list(node2features.values()))
    node_spans.set_yscale("log")
  if len(edge2features.values()):
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
