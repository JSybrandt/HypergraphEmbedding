# This file impliments the hypergraph 2 vec model wherein similarities are
# weighted by each node / community's radius in algebraic distance

from .hypergraph_util import *
from .embedding import EmbedAlgebraicDistance
import numpy as np
import scipy as sp
from scipy.spatial.distance import euclidean
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import logging

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
  return result


################################################################################
# Precompute Observed Probabilities                                            #
################################################################################


def _weight_by_radius(alpha, radius):
  """
  calculates: (α + (1−α)(1−𝑟))
  where α and 𝑟 are in the 0-1 interval.
  α - damping factor. Even the worst radius still contributes this much
  𝑟 - radius in 0-1 scale. Larger represents a less important edge here
  """
  return (alpha + (1 - alpha) * (1 - radius))


## Same Type Probabilities #####################################################


def _init_same_type_probability(idx2neigh, alpha, neigh2radius):
  _shared_info.clear()
  assert alpha >= 0
  assert alpha <= 1
  _shared_info["idx2neigh"] = idx2neigh
  _shared_info["alpha"] = alpha
  _shared_info["neigh2radius"] = neigh2radius


def _same_type_probability(ab, idx2neigh=None, alpha=None, neigh2radius=None):
  if idx2neigh is None:
    idx2neigh = _shared_info["idx2neigh"]
  if alpha is None:
    alpha = _shared_info["alpha"]
  if neigh2radius is None:
    neigh2radius = _shared_info["neigh2radius"]

  a, b = ab
  # A is always 1 to A
  if a == b:
    return 1
  a_neigh = set(idx2neigh[a, :].nonzero()[1])
  b_neigh = set(idx2neigh[b, :].nonzero()[1])
  # Calculate weighted intersection
  numerator = 0
  for neigh_idx in a_neigh.intersection(b_neigh):
    numerator += _weight_by_radius(alpha, neigh2radius[neigh_idx])
  if numerator == 0:
    return 0
  # Calculate weighted union
  denominator = 0
  for neigh_idx in a_neigh.union(b_neigh):
    denominator += _weight_by_radius(alpha, neigh2radius[neigh_idx])
  if denominator == 0:
    return 0
  return numerator / denominator


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
    edge2radius):
  _shared_info.clear()
  _shared_info["node2edge"] = node2edge
  _shared_info["edge2node"] = edge2node
  _shared_info["node2node"] = node2node
  _shared_info["edge2edge"] = edge2edge
  _shared_info["alpha"] = alpha
  _shared_info["node2radius"] = node2radius
  _shared_info["edge2radius"] = edge2radius


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


def PrecomputeWeightedSimilarities(
    hypergraph,
    num_neighbors,
    samples_per_node,
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
      inputs=[left_node_idx,
              right_node_idx,
              left_edge_idx,
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
