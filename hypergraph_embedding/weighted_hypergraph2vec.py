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
