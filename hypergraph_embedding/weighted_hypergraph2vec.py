# This file impliments the hypergraph 2 vec model wherein similarities are
# weighted by each node / community's span in algebraic distance

from . import HypergraphEmbedding
from .hypergraph_util import *
from .algebraic_distance import EmbedAlgebraicDistance
import numpy as np
import scipy as sp
from scipy.spatial.distance import minkowski
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import logging
from random import sample
from collections import namedtuple
from statistics import stdev

import keras
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Multiply, Concatenate
from keras.layers import Dot, Flatten, Average

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

log = logging.getLogger()

_shared_info = {}

################################################################################
# ComputeSpans & Helper functions                                    #
################################################################################


def _init_compute_span(idx2neighbors, idx_emb, neigh_emb):
  _shared_info.clear()
  _shared_info["idx2neighbors"] = idx2neighbors
  _shared_info["idx_emb"] = idx_emb
  _shared_info["neigh_emb"] = neigh_emb


def _compute_span(idx, idx2neighbors=None, idx_emb=None, neigh_emb=None):
  if idx2neighbors is None:
    idx2neighbors = _shared_info["idx2neighbors"]
  if idx_emb is None:
    idx_emb = _shared_info["idx_emb"]
  if neigh_emb is None:
    neigh_emb = _shared_info["neigh_emb"]

  span_less_than = 0
  span_greater_than = 0
  if idx in idx_emb and idx < idx2neighbors.shape[0]:
    my_emb = idx_emb[idx].values
    for neigh_idx in idx2neighbors[idx, :].nonzero()[1]:
      if neigh_idx in neigh_emb:
        diff = np.subtract(neigh_emb[neigh_idx].values, my_emb)
        # Look for values that occur before the current node
        span_less_than = min(span_less_than, min(diff))
        # Look for values that occur _after_ the current node
        span_greater_than = max(span_greater_than, max(diff))
  return idx, span_greater_than - span_less_than


def ComputeSpans(
    hypergraph,
    embedding=None,
    run_in_parallel=True,
    disable_pbar=False):
  """
  Computes the span of each node / edge in the provided embedding.
  Radius is defined as the L2 norm of the distance between an entity's
  embedding and its furthest first-order neighbor.
  For instance, if a node is placed 2 units away from a community, then
  its span will be at least 2.

  inputs:
    - hypergraph: A hypergraph proto message
    - embedding: an optional pre-computed embedding, needed for tests.
                 if not supplied, performs Algebraic Distance in 3d
  outputs:
    (node2span, edge2span): a tuple of dictionary's that maps each
                            node/edge idx to a float span
  """

  if embedding is None:
    embedding = EmbedAlgebraicDistance(
        hypergraph,
        dimension=10,
        iterations=20,
        run_in_parallel=run_in_parallel,
        disable_pbar=disable_pbar)

  assert set(hypergraph.node) == set(embedding.node)
  assert set(hypergraph.edge) == set(embedding.edge)

  workers = multiprocessing.cpu_count() if run_in_parallel else 1

  log.info("Computing span per node wrt edge %s", embedding.method_name)

  node2edge = ToCsrMatrix(hypergraph)
  node2span = {}
  with Pool(workers,
            initializer=_init_compute_span,
            initargs=(node2edge,
                      embedding.node,
                      embedding.edge)) as pool:
    with tqdm(total=len(hypergraph.node), disable=disable_pbar) as pbar:
      for node_idx, span in pool.imap(_compute_span, hypergraph.node):
        node2span[node_idx] = span
        pbar.update(1)

  log.info("Computing span per edge wrt node %s", embedding.method_name)
  edge2node = ToEdgeCsrMatrix(hypergraph)
  edge2span = {}
  with Pool(workers,
            initializer=_init_compute_span,
            initargs=(edge2node,
                      embedding.edge,
                      embedding.node)) as pool:
    with tqdm(total=len(hypergraph.edge), disable=disable_pbar) as pbar:
      for edge_idx, span in pool.imap(_compute_span, hypergraph.edge):
        edge2span[edge_idx] = span
        pbar.update(1)
  return node2span, edge2span


################################################################################
# Zero One Scale - Needs to scale spans                                        #
################################################################################


def _init_zero_one_scale_key(idx2value, min_val, delta_val):
  _shared_info.clear()
  _shared_info["idx2value"] = idx2value
  _shared_info["min_val"] = min_val
  _shared_info["delta_val"] = delta_val


def _zero_one_scale_key(idx, idx2value=None, min_val=None, delta_val=None):
  if idx2value is None:
    idx2value = _shared_info["idx2value"]
  if min_val is None:
    min_val = _shared_info["min_val"]
  if delta_val is None:
    delta_val = _shared_info["delta_val"]
  if delta_val == 0:
    return idx, 1
  else:
    return idx, (idx2value[idx] - min_val) / delta_val


def ZeroOneScaleKeys(idx2value, run_in_parallel=True, disable_pbar=False):
  """
    Scales input dict idx2value to the 0-1 interval. If only one value,
    return 1.
  """
  if len(idx2value) == 0:
    return {}
  workers = multiprocessing.cpu_count() if run_in_parallel else 1
  result = {}
  min_val = min(idx2value.values())
  max_val = max(idx2value.values())
  delta_val = max_val - min_val
  with Pool(workers,
            initializer=_init_zero_one_scale_key,
            initargs=(idx2value,
                      min_val,
                      delta_val)) as pool:
    with tqdm(total=len(idx2value), disable=disable_pbar) as pbar:
      for idx, val in pool.imap(_zero_one_scale_key, idx2value):
        result[idx] = val
        pbar.update(1)
  return result


################################################################################
# Precompute Observed Probabilities                                            #
################################################################################

WeightedSimilarityRecord = namedtuple(
    "WeightedSimilarityRecord",
    (
        "left_node_idx",
        "left_edge_idx",
        "right_node_idx",
        "right_edge_idx",
        "left_weight",
        "right_weight",
        "neighbor_node_indices",
        "neighbor_node_weights",
        "neighbor_edge_indices",
        "neighbor_edge_weights",
        "node_node_prob",
        "edge_edge_prob",
        "node_edge_prob"))
# Set all field defaults to none
WeightedSimilarityRecord.__new__.__defaults__ = \
    (None,) * len(WeightedSimilarityRecord._fields)


def IncOrZero(x):
  if x is None:
    return 0
  else:
    return x + 1


def ValOrZero(x):
  if x is None:
    return 0
  else:
    return x


def PadWithZeros(arrOrNone, idx):
  if arrOrNone is None or idx >= len(arrOrNone):
    return 0
  return arrOrNone[idx]


def _weighted_similarity_records_to_model_input(records, num_neighbors):
  "Converts the above named tuple into (input arrays, output arrays)"
  left_node_idx = []
  right_node_idx = []
  left_edge_idx = []
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

  for r in records:
    left_node_idx.append(IncOrZero(r.left_node_idx))
    right_node_idx.append(IncOrZero(r.right_node_idx))
    left_edge_idx.append(IncOrZero(r.left_edge_idx))
    right_edge_idx.append(IncOrZero(r.right_edge_idx))
    left_weight.append(ValOrZero(r.left_weight))  # if not supplied, set to bad
    right_weight.append(ValOrZero(r.right_weight))
    for i in range(num_neighbors):
      neighbor_node_indices[i].append(PadWithZeros(r.neighbor_node_indices, i))
      neighbor_node_weights[i].append(PadWithZeros(r.neighbor_node_weights, i))
      neighbor_edge_indices[i].append(PadWithZeros(r.neighbor_edge_indices, i))
      neighbor_edge_weights[i].append(PadWithZeros(r.neighbor_edge_weights, i))
    node_node_prob.append(ValOrZero(r.node_node_prob))
    edge_edge_prob.append(ValOrZero(r.edge_edge_prob))
    node_edge_prob.append(ValOrZero(r.node_edge_prob))

  return (
      [left_node_idx,
       left_edge_idx,
       right_node_idx,
       right_edge_idx,
       left_weight,
       right_weight] \
      + neighbor_node_indices \
      + neighbor_node_weights \
      + neighbor_edge_indices \
      + neighbor_edge_weights,
      [node_node_prob,
       edge_edge_prob,
       node_node_prob])


def _weight_by_span(alpha, span):
  """
  calculates: (α + (1−α)(1−𝑟))
  where α and 𝑟 are in the 0-1 interval.
  α - damping factor. Even the worst span still contributes this much
  𝑟 - span in 0-1 scale. Larger represents a less important edge here
  """
  return (alpha + (1 - alpha) * (1 - span))


## Same Type Probabilities #####################################################


def _init_same_type_probability(idx2neigh, alpha, a2span, b2span, is_edge):
  _shared_info.clear()
  assert alpha >= 0
  assert alpha <= 1
  _shared_info["idx2neigh"] = idx2neigh
  _shared_info["alpha"] = alpha
  _shared_info["a2span"] = a2span
  _shared_info["b2span"] = b2span
  _shared_info["is_edge"] = is_edge


def _same_type_probability(ij, idx2neigh=None, alpha=None, b2span=None):
  if idx2neigh is None:
    idx2neigh = _shared_info["idx2neigh"]
  if alpha is None:
    alpha = _shared_info["alpha"]
  if b2span is None:
    b2span = _shared_info["b2span"]

  i, j = ij
  # A is always 1 to A
  if i == j:
    return 1
  a_neigh = set(idx2neigh[i, :].nonzero()[1])
  b_neigh = set(idx2neigh[j, :].nonzero()[1])
  # Calculate weighted intersection
  numerator = 0
  for neigh_idx in a_neigh.intersection(b_neigh):
    numerator += _weight_by_span(alpha, b2span[neigh_idx])
  if numerator == 0:
    return 0
  # Calculate weighted union
  denominator = 0
  for neigh_idx in a_neigh.union(b_neigh):
    denominator += _weight_by_span(alpha, b2span[neigh_idx])
  if denominator == 0:
    return 0
  return numerator / denominator


def _same_type_sample(ij, a2span=None, is_edge=None, alpha=None):
  if a2span is None:
    a2span = _shared_info["a2span"]
  if is_edge is None:
    is_edge = _shared_info["is_edge"]
  if alpha is None:
    alpha = _shared_info["alpha"]

  i, j = ij
  if is_edge:
    return WeightedSimilarityRecord(
        left_edge_idx=i,
        right_edge_idx=j,
        left_weight=_weight_by_span(alpha, a2span[i]),
        right_weight=_weight_by_span(alpha, a2span[j]),
        edge_edge_prob=_same_type_probability(ij))
  else:
    return WeightedSimilarityRecord(
        left_node_idx=i,
        right_node_idx=j,
        left_weight=_weight_by_span(alpha, a2span[i]),
        right_weight=_weight_by_span(alpha, a2span[j]),
        node_node_prob=_same_type_probability(ij))


## Different Type Probabilities ################################################


def _helper_diff_type_prob(a_idx, b_idx, a2b, b2a, a2a, alpha, a2span, b2span):
  "Computes half of the node-edge probability function"
  a_second_neighbors = set(a2a[a_idx, :].nonzero()[1])
  b_neighbors = set(b2a[b_idx, :].nonzero()[1])
  numerator = 0
  denominator = 0
  # Computes weighted probability for all intersecting entities
  for neigh_idx in a_second_neighbors.intersection(b_neighbors):
    prob = _same_type_probability((a_idx, neigh_idx), a2b, alpha, b2span)
    prob *= _weight_by_span(alpha, a2span[neigh_idx])
    numerator += prob
    denominator += prob
  if numerator == 0:
    return 0
  # Computes weighted probability for all union entities not covered by above
  # Note that we only have to compute a_second_neighbors - b_neighbors because
  # by definition the same_type_probability of b_neighbors - a_second_neighbors
  # to a_idx is zero (no shared edges)
  for neigh_idx in a_second_neighbors - b_neighbors:
    prob = _same_type_probability((a_idx, neigh_idx), a2b, alpha, b2span)
    prob *= _weight_by_span(alpha, a2span[neigh_idx])
    denominator += prob
  if denominator == 0:
    return 0
  return numerator / denominator


def _init_node_edge_probability(
    node2edge,
    edge2node,
    node2node,
    edge2edge,
    alpha,
    node2span,
    edge2span,
    num_neighbors):
  _shared_info.clear()
  _shared_info["node2edge"] = node2edge
  _shared_info["edge2node"] = edge2node
  _shared_info["node2node"] = node2node
  _shared_info["edge2edge"] = edge2edge
  _shared_info["alpha"] = alpha
  _shared_info["node2span"] = node2span
  _shared_info["edge2span"] = edge2span
  _shared_info["num_neighbors"] = num_neighbors


def _node_edge_probability(
    node_edge,
    node2edge=None,
    edge2node=None,
    node2node=None,
    edge2edge=None,
    alpha=None,
    node2span=None,
    edge2span=None):
  if node2edge is None:
    node2edge = _shared_info["node2edge"]
  if edge2node is None:
    edge2node = _shared_info["edge2node"]
  if node2node is None:
    node2node = _shared_info["node2node"]
  if edge2edge is None:
    edge2edge = _shared_info["edge2edge"]
  if alpha is None:
    alpha = _shared_info["alpha"]
  if node2span is None:
    node2span = _shared_info["node2span"]
  if edge2span is None:
    edge2span = _shared_info["edge2span"]

  node_idx, edge_idx = node_edge
  prob_node_edge = _helper_diff_type_prob(
      node_idx,
      edge_idx,
      node2edge,
      edge2node,
      node2node,
      alpha,
      node2span,
      edge2span)
  prob_edge_node = _helper_diff_type_prob(
      edge_idx,
      node_idx,
      edge2node,
      node2edge,
      edge2edge,
      alpha,
      edge2span,
      node2span)
  return (prob_node_edge + prob_edge_node) / 2


def _node_edge_sample(
    node_edge,
    num_neighbors=None,
    node2edge=None,
    edge2node=None,
    node2span=None,
    edge2span=None,
    alpha=None):
  "_init_node_edge_prob must have been run"
  if num_neighbors is None:
    num_neighbors = _shared_info["num_neighbors"]
  if node2edge is None:
    node2edge = _shared_info["node2edge"]
  if edge2node is None:
    edge2node = _shared_info["edge2node"]
  if node2span is None:
    node2span = _shared_info["node2span"]
  if edge2span is None:
    edge2span = _shared_info["edge2span"]
  if alpha is None:
    alpha = _shared_info["alpha"]

  node_idx, edge_idx = node_edge
  node_neighbors = list(edge2node[edge_idx, :].nonzero()[1])
  edge_neighbors = list(node2edge[node_idx, :].nonzero()[1])
  node_neighbors_sample = sample(
      node_neighbors,
      min(len(node_neighbors),
          num_neighbors))
  edge_neighbors_sample = sample(
      edge_neighbors,
      min(len(edge_neighbors),
          num_neighbors))
  node_weights = [_weight_by_span(alpha, node2span[n]) for n in node_neighbors_sample]
  edge_weights = [_weight_by_span(alpha, edge2span[e]) for e in edge_neighbors_sample]

  node_edge_prob = _node_edge_probability((node_idx, edge_idx))

  return WeightedSimilarityRecord(
      left_node_idx=node_idx,
      right_edge_idx=edge_idx,
      left_weight=_weight_by_span(alpha, node2span[node_idx]),
      right_weight=_weight_by_span(alpha, edge2span[edge_idx]),
      neighbor_node_indices=node_neighbors_sample,
      neighbor_node_weights=node_weights,
      neighbor_edge_indices=edge_neighbors_sample,
      neighbor_edge_weights=edge_weights,
      node_edge_prob=node_edge_prob)


################################################################################
# Collect Samples and Compute Observed Probabilities                           #
################################################################################


def _sample_per_row(adj_matrix, num_samples_per_row, flip=False):
  for row_idx in range(adj_matrix.shape[0]):
    cols = list(adj_matrix[row_idx, :].nonzero()[1])
    for col_idx in sample(cols, min(len(cols), num_samples_per_row)):
      if flip:
        yield (col_idx, row_idx)
      else:
        yield (row_idx, col_idx)


def PrecomputeWeightedSimilarities(
    hypergraph,
    num_neighbors,
    samples_per,
    alpha,
    run_in_parallel=True,
    disable_pbar=False):
  """
  Computes node-node, node-edge, and edge-edge similarities weighted by
  algebraic distance.

  𝑃𝑟(𝑛𝑖, 𝑛𝑗) = {∑_{𝑒 ∈ Γ(𝑛𝑖) ∩ Γ(𝑛𝑗)} α + (1−α)(1−𝑟(𝑒))}
               -----------------------------------------
               {∑_{𝑒 ∈ Γ(𝑛𝑖) ∪ Γ(𝑛𝑗)} α + (1−α)(1−𝑟(𝑒))}

  𝑃𝑟(𝑒𝑖, 𝑒𝑗) = {∑_{𝑛 ∈ Γ(𝑒𝑖) ∩ Γ(𝑒𝑗)} α + (1−α)(1−𝑟(𝑛))}
               -----------------------------------------
               {∑_{𝑛 ∈ Γ(𝑒𝑖) ∪ Γ(𝑒𝑗)} α + (1−α)(1−𝑟(𝑛))}

  𝑃𝑟(𝑛, 𝑒) = 1/2 (   ∑_{𝑛'∈ Γ(𝑒) ∩ Γ(Γ(𝑛))} 𝑃𝑟(𝑛, 𝑛')(α + (1−α)(1−𝑟(𝑛')))   )
                 (   ----------------------------------------------------   )
                 (       ∑_{𝑛'∈ Γ(Γ(𝑛))} 𝑃𝑟(𝑛, 𝑛')(α + (1−α)(1−𝑟(𝑛')))      )
                 (                                                          )
                 ( + ∑_{𝑒'∈ Γ(𝑛) ∩ Γ(Γ(𝑒))} 𝑃𝑟(𝑒, 𝑒')(α + (1−α)(1−𝑟(𝑒')))   )
                 (   −−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−   )
                 (       ∑_{𝑛'∈ Γ(Γ(𝑒))} 𝑃𝑟(𝑒, 𝑒')(α + (1−α)(1−𝑟(𝑒')))      )

  """
  num_cores = multiprocessing.cpu_count() if run_in_parallel else 1
  # return value
  similarity_records = []

  log.info("Converting hypergraph to node-major sparse matrix")
  node2edge = ToCsrMatrix(hypergraph)
  log.info("Identifying nodes sharing edges")
  node2node = node2edge * node2edge.T
  log.info("Converting hypergraph to edge-major sparse matrix")
  edge2node = ToEdgeCsrMatrix(hypergraph)
  log.info("Identifying edges sharing nodes")
  edge2edge = edge2node * edge2node.T

  log.info("Calculating algebraic span per point")
  node2span, edge2span = ComputeSpans(
      hypergraph,
      run_in_parallel=run_in_parallel,
      disable_pbar=disable_pbar)

  log.info("Scaling Node Radii")
  node2span = ZeroOneScaleKeys(
      node2span,
      run_in_parallel=run_in_parallel,
      disable_pbar=disable_pbar)

  log.info("Scaling Edge Radii")
  edge2span = ZeroOneScaleKeys(
      edge2span,
      run_in_parallel=run_in_parallel,
      disable_pbar=disable_pbar)

  log.info("Sampling Per Node")
  with Pool(num_cores,
            initializer=_init_same_type_probability,
            initargs=(
              node2edge, #idx2neigh
              alpha, #alpha
              node2span, #a2span
              edge2span, #b2span
              False #is_edge
            )) as pool:
    with tqdm(total=len(hypergraph.node) * samples_per,
              disable=disable_pbar) as pbar:
      for result in pool.imap(_same_type_sample,
                              _sample_per_row(node2node,
                                              num_neighbors)):
        similarity_records.append(result)
        pbar.update(1)

  log.info("Sampling Per Edge")
  with Pool(num_cores,
            initializer=_init_same_type_probability,
            initargs=(
              edge2node, #idx2neigh
              alpha, #alpha
              edge2span, #a2span
              node2span, #b2span
              True #is_edge
            )) as pool:
    with tqdm(total=len(hypergraph.edge) * samples_per,
              disable=disable_pbar) as pbar:
      for result in pool.imap(_same_type_sample,
                              _sample_per_row(edge2edge,
                                              num_neighbors)):
        similarity_records.append(result)
        pbar.update(1)

  # Begin the node-edge pool
  with Pool(num_cores,
            initializer=_init_node_edge_probability,
            initargs=(
              node2edge, #node2edge
              edge2node, #edge2node
              node2node, #node2node
              edge2edge, #edge2edge
              alpha, #alpha
              node2span, #node2span
              edge2span, #edge2span
              num_neighbors
            )) as pool:
    log.info("Identifying all second-order edges per node")
    node2second_edge = node2node * node2edge
    log.info("Sampling second-order edges per node")
    with tqdm(total=len(hypergraph.node) * samples_per,
              disable=disable_pbar) as pbar:
      for result in pool.imap(_node_edge_sample,
                              _sample_per_row(node2second_edge,
                                              num_neighbors)):
        similarity_records.append(result)
        pbar.update(1)

    log.info("Identifying all second-order nodes per edge")
    edge2second_node = edge2edge * edge2node
    log.info("Sampling second-order nodes per edge")
    with tqdm(total=len(hypergraph.edge) * samples_per,
              disable=disable_pbar) as pbar:
      for result in pool.imap(_node_edge_sample,
                              _sample_per_row(edge2second_node,
                                              num_neighbors,
                                              flip=True)):
        similarity_records.append(result)
        pbar.update(1)
  return similarity_records


################################################################################
# Creates the ML Model for the weighted hypergraph2vec method                  #
################################################################################


def GetWeightedModel(hypergraph, dimension, num_neighbors):

  log.info("Constructing Weighted Keras Model")

  left_node_idx = Input((1,), name="left_node_idx", dtype=np.int32)
  right_node_idx = Input((1,), name="right_node_idx", dtype=np.int32)

  left_edge_idx = Input((1,), name="left_edge_idx", dtype=np.int32)
  right_edge_idx = Input((1,), name="right_edge_idx", dtype=np.int32)

  left_weight = Input((1,), name="left_weight", dtype=np.float32)
  right_weight = Input((1,), name="right_weight", dtype=np.float32)

  neighbor_node_indices = [
      Input((1,
            ),
            dtype=np.int32,
            name="neighbor_node_idx_{}".format(i))
      for i in range(num_neighbors)
  ]

  neighbor_node_weights = [
      Input((1,
            ),
            dtype=np.float32,
            name="neighbor_node_weight_{}".format(i))
      for i in range(num_neighbors)
  ]

  neighbor_edge_indices = [
      Input((1,
            ),
            dtype=np.int32,
            name="neighbor_edge_idx_{}".format(i))
      for i in range(num_neighbors)
  ]

  neighbor_edge_weights = [
      Input((1,
            ),
            dtype=np.float32,
            name="neighbor_edge_weight_{}".format(i))
      for i in range(num_neighbors)
  ]

  # Gets domain of nodes / edges
  max_node_idx = max([i for i in hypergraph.node])
  max_edge_idx = max([i for i in hypergraph.edge])

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

  left_node_vec = Flatten(name="left_node_vec")(node_emb(left_node_idx))
  left_edge_vec = Flatten(name="left_edge_vec")(edge_emb(left_edge_idx))
  right_node_vec = Flatten(name="right_node_vec")(node_emb(right_node_idx))
  right_edge_vec = Flatten(name="right_edge_vec")(edge_emb(right_edge_idx))

  neighbor_node_vecs = [
      Flatten()(node_emb(neighbor_node_idx))
      for neighbor_node_idx in neighbor_node_indices
  ]

  neighbor_edge_vecs = [
      Flatten()(edge_emb(neighbor_edge_idx))
      for neighbor_edge_idx in neighbor_edge_indices
  ]

  node_node_prediction = Dense(
      1,
      activation="sigmoid",
      name="node_node_prob")(
          Concatenate(1)(
              [Dot(1)([left_node_vec,
                       right_node_vec]),
               left_weight,
               right_weight]))

  edge_edge_prediction = Dense(
      1,
      activation="sigmoid",
      name="edge_edge_prob")(
          Concatenate(1)(
              [Dot(1)([left_edge_vec,
                       right_edge_vec]),
               left_weight,
               right_weight]))

  node_neighbor_dot_sigs = [
      Dense(1,
            activation="sigmoid")(
                Concatenate(1)(
                    [Dot(1)([n_vec,
                             left_node_vec]),
                     left_weight,
                     n_weight])) for n_vec,
      n_weight in zip(neighbor_node_vecs,
                    neighbor_node_weights)
  ]
  node_neighbor_avg = Average()(node_neighbor_dot_sigs)

  edge_neighbor_dot_sigs = [
      Dense(1,
            activation="sigmoid")(
                Concatenate(1)(
                    [Dot(1)([n_vec,
                             right_edge_vec]),
                     right_weight,
                     n_weight])) for n_vec,
      n_weight in zip(neighbor_edge_vecs,
                    neighbor_edge_weights)
  ]
  edge_neighbor_avg = Average()(edge_neighbor_dot_sigs)

  node_edge_prediction = Multiply(name="node_edge_prob")(
      [node_neighbor_avg,
       edge_neighbor_avg])

  model = Model(
      # THESE MUST LINE UP EXACTLY WITH
      # _weighted_similarity_records_to_model_input
      inputs=[left_node_idx,
              left_edge_idx,
              right_node_idx,
              right_edge_idx,
              left_weight,
              right_weight] \
             + neighbor_node_indices \
             + neighbor_node_weights \
             + neighbor_edge_indices \
             + neighbor_edge_weights,
      outputs=[node_node_prediction,
               edge_edge_prediction,
               node_edge_prediction])
  model.compile(optimizer="adagrad", loss="kullback_leibler_divergence")
  return model


################################################################################
# Hook in for runner                                                           #
################################################################################


def EmbedWeightedHypergraph(
    hypergraph,
    dimension,
    num_neighbors=10,
    alpha=0.25,
    samples_per=250,
    batch_size=256,
    epochs=5,
    disable_pbar=False,
    debug_summary_path=None):
  similarity_records = PrecomputeWeightedSimilarities(
      hypergraph,
      num_neighbors,
      samples_per,
      alpha,
      disable_pbar=disable_pbar)

  if debug_summary_path is not None:
    WriteDebugSummary(debug_summary_path, similarity_records)

  input_features, output_probs = _weighted_similarity_records_to_model_input(
    similarity_records,
    num_neighbors)

  model = GetWeightedModel(hypergraph, dimension, num_neighbors)
  model.fit(input_features, output_probs, batch_size=batch_size, epochs=epochs)

  log.info("Recording Embeddings")

  node_weights = model.get_layer("node_embedding").get_weights()[0]
  edge_weights = model.get_layer("edge_embedding").get_weights()[0]

  embedding = HypergraphEmbedding()
  embedding.dim = dimension
  embedding.method_name = "WeightedHypergraph"

  for node_idx in hypergraph.node:
    embedding.node[node_idx].values.extend(node_weights[node_idx + 1])
  for edge_idx in hypergraph.edge:
    embedding.edge[edge_idx].values.extend(edge_weights[edge_idx + 1])
  return embedding


def _log_distribution_info(name, distribution):
  log.info(name)
  log.info(" > Size : %i", len(distribution))
  log.info(" > Range: %f - %f", min(distribution), max(distribution))
  log.info(" > Mean : %f", sum(distribution) / len(distribution))
  log.info(" > Std. : %f", stdev(distribution))


def WriteDebugSummary(debug_summary_path, sim_records):

  log.info("Writing Debug Summary to %s", debug_summary_path)
  node2span = {}
  edge2span = {}
  for r in sim_records:
    if r.left_node_idx is not None and r.left_node_idx not in node2span:
      node2span[r.left_node_idx] = r.left_weight
    if r.right_node_idx is not None and r.right_node_idx not in node2span:
      node2span[r.right_node_idx] = r.right_weight
    if r.left_edge_idx is not None and r.left_edge_idx not in edge2span:
      edge2span[r.left_edge_idx] = r.left_weight
    if r.right_edge_idx is not None and r.right_edge_idx not in edge2span:
      edge2span[r.right_edge_idx] = r.right_weight

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
  node_spans.set_title("Node Spans")
  node_spans.hist(list(node2span.values()))
  node_spans.set_yscale("log")
  edge_spans.set_title("Edge Spans")
  edge_spans.hist(list(edge2span.values()))
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
  _log_distribution_info("NodeSpans", list(node2span.values()))
  _log_distribution_info("EdgeSpans", list(edge2span.values()))
  _log_distribution_info("Node-Node Prob.", nn_probs)
  _log_distribution_info("Edge-Edge Prob.", ee_probs)
  _log_distribution_info("Node-Edge Prob.", ne_probs)
