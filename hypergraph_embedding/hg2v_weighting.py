################################################################################
# Hypergraph 2 Vec Weighting Schemes                                           #
# This module is responsible for computing weights for hypergraph node-edges   #
# Each function maps a hypergraph to two dictionaries, node2weight and         #
# edge2weight. These can then be used with hg2v_sample.                        #
################################################################################

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

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

log = logging.getLogger()

_shared_info = {}


def WeightByNeighborhood(hypergraph, alpha):
  "The goal is that larger neighborhoods contribute less"
  log.info("Getting neighboorhood sizes for all nodes / edges")
  node2weight = {idx: len(node.edges) for idx, node in hypergraph.node.items()}
  edge2weight = {idx: len(edge.nodes) for idx, edge in hypergraph.edge.items()}

  log.info("Zero one scaling")
  node2weight = ZeroOneScaleKeys(node2weight)
  edge2weight = ZeroOneScaleKeys(edge2weight)

  log.info("1-value")
  node2weight = OneMinusValues(node2weight)
  edge2weight = OneMinusValues(edge2weight)

  log.info("Alpha scaling")
  node2weight = AlphaScaleValues(node2weight, alpha)
  edge2weight = AlphaScaleValues(edge2weight, alpha)

  return node2weight, edge2weight


def WeightByAlgebraicSpan(hypergraph, alpha):
  node2weight, edge2weight = ComputeSpans(hypergraph)

  log.info("Zero one scaling")
  node2weight = ZeroOneScaleKeys(node2weight)
  edge2weight = ZeroOneScaleKeys(edge2weight)

  log.info("1-value")
  node2weight = OneMinusValues(node2weight)
  edge2weight = OneMinusValues(edge2weight)

  log.info("Alpha scaling")
  node2weight = AlphaScaleValues(node2weight, alpha)
  edge2weight = AlphaScaleValues(edge2weight, alpha)

  return node2weight, edge2weight


def UniformWeight(hypergraph):
  node2weight = {node_idx: 1 for node_idx in hypergraph.node}
  edge2weight = {edge_idx: 1 for edge_idx in hypergraph.edge}
  return node2weight, edge2weight


################################################################################
# ComputeSpans & Helper functions                                              #
# This computes the maximum spread of a node/edge wrt algebraic distance       #
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
        dimension=5,
        iterations=10,
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
# Zero One Scale - We may need to scale each weight                            #
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
# Weighting Util Functions                                                     #
################################################################################


def OneMinusValues(data):
  return {k: 1 - v for k, v in data.items()}


def AlphaScaleValues(data, alpha):
  "Alpha is a minimum support for a value"
  assert alpha >= 0
  assert alpha <= 1
  return {k: (alpha + (1 - alpha) * v) for k, v in data.items()}
