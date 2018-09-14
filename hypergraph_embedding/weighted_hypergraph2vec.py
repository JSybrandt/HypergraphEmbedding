# This file impliments the hypergraph 2 vec model wherein similarities are
# weighted by each node / community's radius in algebraic distance

from . import HypergraphEmbedding
from .hypergraph_util import *
from .algebraic_distance import EmbedAlgebraicDistance
import numpy as np
import scipy as sp
from scipy.spatial.distance import euclidean
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import logging
from random import sample
from collections import namedtuple

import keras
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Multiply, Concatenate
from keras.layers import Dot, Flatten, Average

log = logging.getLogger()

_shared_info = {}

################################################################################
# ComputeAlgebraicRadius & Helper functions                                    #
################################################################################


def _init_compute_radius(idx2neighbors, idx_emb, neigh_emb):
  _shared_info.clear()
  _shared_info["idx2neighbors"] = idx2neighbors
  _shared_info["idx_emb"] = idx_emb
  _shared_info["neigh_emb"] = neigh_emb


def _compute_radius(idx, idx2neighbors=None, idx_emb=None, neigh_emb=None):
  if idx2neighbors is None:
    idx2neighbors = _shared_info["idx2neighbors"]
  if idx_emb is None:
    idx_emb = _shared_info["idx_emb"]
  if neigh_emb is None:
    neigh_emb = _shared_info["neigh_emb"]

  radius = 0
  if idx in idx_emb and idx < idx2neighbors.shape[0]:
    my_emb = idx_emb[idx].values
    for neigh_idx in idx2neighbors[idx, :].nonzero()[1]:
      if neigh_idx in neigh_emb:
        dist = euclidean(neigh_emb[neigh_idx].values, my_emb)
        if radius < dist:
          radius = dist
  return idx, radius


def ComputeAlgebraicRadius(
    hypergraph,
    embedding=None,
    run_in_parallel=True,
    disable_pbar=False):
  """
  Computes the radius of each node / edge in the provided embedding.
  Radius is defined as the L2 norm of the distance between an entity's
  embedding and its furthest first-order neighbor.
  For instance, if a node is placed 2 units away from a community, then
  its radius will be at least 2.

  inputs:
    - hypergraph: A hypergraph proto message
    - embedding: an optional pre-computed embedding, needed for tests.
                 if not supplied, performs Algebraic Distance in 3d
  outputs:
    (node2radius, edge2radus): a tuple of dictionary's that maps each
                               node/edge idx to a float radius
  """

  if embedding is None:
    embedding = EmbedAlgebraicDistance(
        hypergraph,
        dimension=1,
        iterations=10,
        run_in_parallel=run_in_parallel,
        disable_pbar=disable_pbar)

  assert set(hypergraph.node) == set(embedding.node)
  assert set(hypergraph.edge) == set(embedding.edge)

  workers = multiprocessing.cpu_count() if run_in_parallel else 1

  log.info("Computing radius per node wrt edge %s", embedding.method_name)

  node2edge = ToCsrMatrix(hypergraph)
  node2radius = {}
  with Pool(workers,
            initializer=_init_compute_radius,
            initargs=(node2edge,
                      embedding.node,
                      embedding.edge)) as pool:
    with tqdm(total=len(hypergraph.node), disable=disable_pbar) as pbar:
      for node_idx, radius in pool.imap(_compute_radius, hypergraph.node):
        node2radius[node_idx] = radius
        pbar.update(1)

  log.info("Computing radius per edge wrt node %s", embedding.method_name)
  edge2node = ToEdgeCsrMatrix(hypergraph)
  edge2radius = {}
  with Pool(workers,
            initializer=_init_compute_radius,
            initargs=(edge2node,
                      embedding.edge,
                      embedding.node)) as pool:
    with tqdm(total=len(hypergraph.edge), disable=disable_pbar) as pbar:
      for edge_idx, radius in pool.imap(_compute_radius, hypergraph.edge):
        edge2radius[edge_idx] = radius
        pbar.update(1)
  return node2radius, edge2radius


################################################################################
# Zero One Scale - Needs to scale radii                                        #
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
  max_radius = max(idx2value.values())
  delta_val = max_radius - min_val
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
        "left_radius",
        "right_radius",
        "neighbor_node_indices",
        "neighbor_node_radii",
        "neighbor_edge_indices",
        "neighbor_edge_radii",
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


def ValOrOne(x):
  if x is None:
    return 1
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
  left_radius = []
  right_radius = []
  neighbor_node_indices = [[] for _ in range(num_neighbors)]
  neighbor_node_radii = [[] for _ in range(num_neighbors)]
  neighbor_edge_indices = [[] for _ in range(num_neighbors)]
  neighbor_edge_radii = [[] for _ in range(num_neighbors)]
  node_node_prob = []
  edge_edge_prob = []
  node_edge_prob = []

  for r in records:
    left_node_idx.append(IncOrZero(r.left_node_idx))
    right_node_idx.append(IncOrZero(r.right_node_idx))
    left_edge_idx.append(IncOrZero(r.left_edge_idx))
    right_edge_idx.append(IncOrZero(r.right_edge_idx))
    left_radius.append(ValOrOne(r.left_radius))  # if not supplied, set to bad
    right_radius.append(ValOrOne(r.right_radius))
    for i in range(num_neighbors):
      neighbor_node_indices[i].append(PadWithZeros(r.neighbor_node_indices, i))
      neighbor_node_radii[i].append(PadWithZeros(r.neighbor_node_radii, i))
      neighbor_edge_indices[i].append(PadWithZeros(r.neighbor_edge_indices, i))
      neighbor_edge_radii[i].append(PadWithZeros(r.neighbor_edge_radii, i))
    node_node_prob.append(ValOrZero(r.node_node_prob))
    edge_edge_prob.append(ValOrZero(r.edge_edge_prob))
    node_edge_prob.append(ValOrZero(r.node_edge_prob))

  return (
      [left_node_idx,
       left_edge_idx,
       right_node_idx,
       right_edge_idx,
       left_radius,
       right_radius] \
      + neighbor_node_indices \
      + neighbor_node_radii \
      + neighbor_edge_indices \
      + neighbor_edge_radii,
      [node_node_prob,
       edge_edge_prob,
       node_node_prob])


def _weight_by_radius(alpha, radius):
  """
  calculates: (α + (1−α)(1−𝑟))
  where α and 𝑟 are in the 0-1 interval.
  α - damping factor. Even the worst radius still contributes this much
  𝑟 - radius in 0-1 scale. Larger represents a less important edge here
  """
  return (alpha + (1 - alpha) * (1 - radius))


## Same Type Probabilities #####################################################


def _init_same_type_probability(idx2neigh, alpha, a2radius, b2radius, is_edge):
  _shared_info.clear()
  assert alpha >= 0
  assert alpha <= 1
  _shared_info["idx2neigh"] = idx2neigh
  _shared_info["alpha"] = alpha
  _shared_info["a2radius"] = a2radius
  _shared_info["b2radius"] = b2radius
  _shared_info["is_edge"] = is_edge


def _same_type_probability(ij, idx2neigh=None, alpha=None, b2radius=None):
  if idx2neigh is None:
    idx2neigh = _shared_info["idx2neigh"]
  if alpha is None:
    alpha = _shared_info["alpha"]
  if b2radius is None:
    b2radius = _shared_info["b2radius"]

  i, j = ij
  # A is always 1 to A
  if i == j:
    return 1
  a_neigh = set(idx2neigh[i, :].nonzero()[1])
  b_neigh = set(idx2neigh[j, :].nonzero()[1])
  # Calculate weighted intersection
  numerator = 0
  for neigh_idx in a_neigh.intersection(b_neigh):
    numerator += _weight_by_radius(alpha, b2radius[neigh_idx])
  if numerator == 0:
    return 0
  # Calculate weighted union
  denominator = 0
  for neigh_idx in a_neigh.union(b_neigh):
    denominator += _weight_by_radius(alpha, b2radius[neigh_idx])
  if denominator == 0:
    return 0
  return numerator / denominator


def _same_type_sample(ij, a2radius=None, is_edge=None):
  if a2radius is None:
    a2radius = _shared_info["a2radius"]
  if is_edge is None:
    is_edge = _shared_info["is_edge"]
  i, j = ij
  if is_edge:
    return WeightedSimilarityRecord(
        left_edge_idx=i,
        right_edge_idx=j,
        left_radius=a2radius[i],
        right_radius=a2radius[j],
        edge_edge_prob=_same_type_probability(ij))
  else:
    return WeightedSimilarityRecord(
        left_node_idx=i,
        right_node_idx=j,
        left_radius=a2radius[i],
        right_radius=a2radius[j],
        node_node_prob=_same_type_probability(ij))


## Different Type Probabilities ################################################


def _helper_diff_type_prob(
    a_idx,
    b_idx,
    a2b,
    b2a,
    a2a,
    alpha,
    a2radius,
    b2radius):
  "Computes half of the node-edge probability function"
  a_second_neighbors = set(a2a[a_idx, :].nonzero()[1])
  b_neighbors = set(b2a[b_idx, :].nonzero()[1])
  numerator = 0
  denominator = 0
  # Computes weighted probability for all intersecting entities
  for neigh_idx in a_second_neighbors.intersection(b_neighbors):
    prob = _same_type_probability((a_idx, neigh_idx), a2b, alpha, b2radius)
    prob *= _weight_by_radius(alpha, a2radius[neigh_idx])
    numerator += prob
    denominator += prob
  if numerator == 0:
    return 0
  # Computes weighted probability for all union entities not covered by above
  # Note that we only have to compute a_second_neighbors - b_neighbors because
  # by definition the same_type_probability of b_neighbors - a_second_neighbors
  # to a_idx is zero (no shared edges)
  for neigh_idx in a_second_neighbors - b_neighbors:
    prob = _same_type_probability((a_idx, neigh_idx), a2b, alpha, b2radius)
    prob *= _weight_by_radius(alpha, a2radius[neigh_idx])
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
    node2radius,
    edge2radius,
    num_neighbors):
  _shared_info.clear()
  _shared_info["node2edge"] = node2edge
  _shared_info["edge2node"] = edge2node
  _shared_info["node2node"] = node2node
  _shared_info["edge2edge"] = edge2edge
  _shared_info["alpha"] = alpha
  _shared_info["node2radius"] = node2radius
  _shared_info["edge2radius"] = edge2radius
  _shared_info["num_neighbors"] = num_neighbors


def _node_edge_probability(
    node_edge,
    node2edge=None,
    edge2node=None,
    node2node=None,
    edge2edge=None,
    alpha=None,
    node2radius=None,
    edge2radius=None):
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
  if node2radius is None:
    node2radius = _shared_info["node2radius"]
  if edge2radius is None:
    edge2radius = _shared_info["edge2radius"]

  node_idx, edge_idx = node_edge
  prob_node_edge = _helper_diff_type_prob(
      node_idx,
      edge_idx,
      node2edge,
      edge2node,
      node2node,
      alpha,
      node2radius,
      edge2radius)
  prob_edge_node = _helper_diff_type_prob(
      edge_idx,
      node_idx,
      edge2node,
      node2edge,
      edge2edge,
      alpha,
      edge2radius,
      node2radius)
  return (prob_node_edge + prob_edge_node) / 2


def _node_edge_sample(
    node_edge,
    num_neighbors=None,
    node2edge=None,
    edge2node=None,
    node2radius=None,
    edge2radius=None):
  "_init_node_edge_prob must have been run"
  if num_neighbors is None:
    num_neighbors = _shared_info["num_neighbors"]
  if node2edge is None:
    node2edge = _shared_info["node2edge"]
  if edge2node is None:
    edge2node = _shared_info["edge2node"]
  if node2radius is None:
    node2radius = _shared_info["node2radius"]
  if edge2radius is None:
    edge2radius = _shared_info["edge2radius"]

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
  node_radii = [node2radius[n] for n in node_neighbors_sample]
  edge_radii = [edge2radius[e] for e in edge_neighbors_sample]

  node_edge_prob = _node_edge_probability((node_idx, edge_idx))

  return WeightedSimilarityRecord(
      left_node_idx=node_idx,
      right_edge_idx=edge_idx,
      left_radius=node2radius[node_idx],
      right_radius=edge2radius[edge_idx],
      neighbor_node_indices=node_neighbors_sample,
      neighbor_node_radii=node_radii,
      neighbor_edge_indices=edge_neighbors_sample,
      neighbor_edge_radii=edge_radii)


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

  log.info("Calculating algebraic radius per point")
  node2radius, edge2radius = ComputeAlgebraicRadius(
      hypergraph,
      run_in_parallel=run_in_parallel,
      disable_pbar=disable_pbar)

  log.info("Scaling Node Radii")
  node2radius = ZeroOneScaleKeys(
      node2radius,
      run_in_parallel=run_in_parallel,
      disable_pbar=disable_pbar)

  log.info("Scaling Edge Radii")
  edge2radius = ZeroOneScaleKeys(
      edge2radius,
      run_in_parallel=run_in_parallel,
      disable_pbar=disable_pbar)

  log.info("Sampling Per Node")
  with Pool(num_cores,
            initializer=_init_same_type_probability,
            initargs=(
              node2edge, #idx2neigh
              alpha, #alpha
              node2radius, #a2radius
              edge2radius, #b2radius
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
              edge2radius, #a2radius
              node2radius, #b2radius
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
              node2radius, #node2radius
              edge2radius, #edge2radius
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

  return _weighted_similarity_records_to_model_input(
      similarity_records,
      num_neighbors)


################################################################################
# Creates the ML Model for the weighted hypergraph2vec method                  #
################################################################################


def GetWeightedModel(hypergraph, dimension, num_neighbors):

  log.info("Constructing Weighted Keras Model")

  left_node_idx = Input((1,), name="left_node_idx", dtype=np.int32)
  right_node_idx = Input((1,), name="right_node_idx", dtype=np.int32)

  left_edge_idx = Input((1,), name="left_edge_idx", dtype=np.int32)
  right_edge_idx = Input((1,), name="right_edge_idx", dtype=np.int32)

  left_radius = Input((1,), name="left_radius", dtype=np.float32)
  right_radius = Input((1,), name="right_radius", dtype=np.float32)

  neighbor_edge_indices = [
      Input((1,
            ),
            dtype=np.int32,
            name="neighbor_edge_idx_{}".format(i))
      for i in range(num_neighbors)
  ]
  neighbor_edge_radii = [
      Input((1,
            ),
            dtype=np.float32,
            name="neighbor_edge_radius_{}".format(i))
      for i in range(num_neighbors)
  ]

  neighbor_node_indices = [
      Input((1,
            ),
            dtype=np.int32,
            name="neighbor_node_idx_{}".format(i))
      for i in range(num_neighbors)
  ]
  neighbor_node_radii = [
      Input((1,
            ),
            dtype=np.float32,
            name="neighbor_node_radius_{}".format(i))
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

  node_node_prob = Dense(
      1,
      activation="sigmoid",
      input_shape=(3,
                  ),
      name="node_node_prob")
  edge_edge_prob = Dense(
      1,
      activation="sigmoid",
      input_shape=(3,
                  ),
      name="edge_edge_prob")

  node_node_prediction = node_node_prob(
      Concatenate(1)(
          [Dot(1)([left_node_vec,
                   right_node_vec]),
           left_radius,
           right_radius]))
  edge_edge_prediction = edge_edge_prob(
      Concatenate(1)(
          [Dot(1)([left_edge_vec,
                   right_edge_vec]),
           left_radius,
           right_radius]))

  node_neighbor_pred = Dense(
      1,
      activation="sigmoid",
      name="node_neighbor_pred")(
          Concatenate(1)([
              node_node_prob(
                  Concatenate(1)(
                      [Dot(1)([n_vec,
                               left_node_vec]),
                       left_radius,
                       n_rad])) for n_vec,
              n_rad in zip(neighbor_node_vecs,
                           neighbor_node_radii)
          ]))
  edge_neighbor_pred = Dense(
      1,
      activation="sigmoid",
      name="edge_neighbor_pred")(
          Concatenate(1)([
              edge_edge_prob(
                  Concatenate(1)(
                      [Dot(1)([n_vec,
                               right_edge_vec]),
                       right_radius,
                       n_rad])) for n_vec,
              n_rad in zip(neighbor_edge_vecs,
                           neighbor_edge_radii)
          ]))
  node_edge_prediction = Dense(
      1,
      activation="sigmoid",
      name="node_edge_prediction")(
          Concatenate(1)([node_neighbor_pred,
                          edge_neighbor_pred]))

  model = Model(
      # THESE MUST LINE UP EXACTLY WITH
      # _weighted_similarity_records_to_model_input
      inputs=[left_node_idx,
              left_edge_idx,
              right_node_idx,
              right_edge_idx,
              left_radius,
              right_radius] \
             + neighbor_node_indices \
             + neighbor_node_radii \
             + neighbor_edge_indices \
             + neighbor_edge_radii,
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
    samples_per=50,
    batch_size=256,
    epochs=5):
  input_features, output_probs = PrecomputeWeightedSimilarities(
      hypergraph,
      num_neighbors,
      samples_per,
      alpha)
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
