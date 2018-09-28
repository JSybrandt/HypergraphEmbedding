from . import HypergraphEmbedding
from .hypergraph_util import *
import numpy as np
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import logging

_shared_info = {}
log = logging.getLogger()

################################################################################
# AlgebraicDistance - Helper and runner                                        #
################################################################################

## Helper functions to update embeddings ######################################


def _init_update_alg_dist(A2B, B2A, A2emb, B2emb):
  _shared_info.clear()

  assert A2B.shape[0] == B2A.shape[1]
  assert A2B.shape[1] == B2A.shape[0]
  assert A2B.shape[0] == A2emb.shape[0]
  assert A2B.shape[1] == B2emb.shape[0]
  assert A2emb.shape[1] == B2emb.shape[1]

  _shared_info["A2B"] = A2B
  _shared_info["B2A"] = B2A
  _shared_info["A2emb"] = A2emb
  _shared_info["B2emb"] = B2emb


def _update_alg_dist(a_idx, A2B=None, B2A=None, A2emb=None, B2emb=None):
  if A2B is None:
    A2B = _shared_info["A2B"]
  if B2A is None:
    B2A = _shared_info["B2A"]
  if A2emb is None:
    A2emb = _shared_info["A2emb"]
  if B2emb is None:
    B2emb = _shared_info["B2emb"]

  a_emb = A2emb[a_idx, :]

  b_emb_weight = [(B2emb[b_idx], 1/B2A[b_idx].nnz)
                  for b_idx in A2B[a_idx].nonzero()[1]]
  b_emb = sum(e * w for e, w in b_emb_weight) / sum(w for _, w in b_emb_weight)

  return a_idx, (a_emb + b_emb) / 2


def _helper_update_embeddings(
    hypergraph,
    node_embeddings,
    edge_embeddings,
    node2edges,
    edge2nodes,
    workers,
    disable_pbar):
  if not disable_pbar:
    log.info("Placing nodes with respect to edges")
  new_node_embeddings = np.copy(node_embeddings)
  with Pool(workers,
            initializer=_init_update_alg_dist,
            initargs=(node2edges,#A2B
                      edge2nodes,#B2A
                      node_embeddings, #A2emb
                      edge_embeddings #B2emb
                      )) as pool:
    with tqdm(total=len(hypergraph.node), disable=disable_pbar) as pbar:
      for idx, emb in pool.imap(_update_alg_dist, hypergraph.node):
        new_node_embeddings[idx, :] = emb
        pbar.update(1)

  if not disable_pbar:
    log.info("Placing edges with respect to nodes")
  new_edge_embeddings = np.copy(edge_embeddings)
  with Pool(workers,
            initializer=_init_update_alg_dist,
            initargs=(edge2nodes,#A2B
                      node2edges,#B2a
                      edge_embeddings, #A2emb
                      new_node_embeddings #B2emb
                      )) as pool:
    with tqdm(total=len(hypergraph.edge), disable=disable_pbar) as pbar:
      for idx, emb in pool.imap(_update_alg_dist, hypergraph.edge):
        new_edge_embeddings[idx, :] = emb
        pbar.update(1)
  return new_node_embeddings, new_edge_embeddings


## Helper functions to scale embeddings ########################################


def _init_scale_alg_dist(embedding, min_embedding, delta_embedding):
  _shared_info.clear()
  assert embedding.shape[1] == len(min_embedding)
  assert embedding.shape[1] == len(delta_embedding)
  _shared_info["embedding"] = embedding
  _shared_info["min_embedding"] = min_embedding
  _shared_info["delta_embedding"] = delta_embedding


def _scale_alg_dist(
    idx,
    embedding=None,
    min_embedding=None,
    delta_embedding=None):
  if embedding is None:
    embedding = _shared_info["embedding"]
  if min_embedding is None:
    min_embedding = _shared_info["min_embedding"]
  if delta_embedding is None:
    delta_embedding = _shared_info["delta_embedding"]
  return idx, (embedding[idx, :] - min_embedding) / delta_embedding


def _helper_scale_embeddings(
    hypergraph,
    node_embeddings,
    edge_embeddings,
    workers,
    disable_pbar):
  if not disable_pbar:
    log.info("Getting min-max embedding per dimension")
  min_edge_embedding = np.min(edge_embeddings, axis=0)
  min_node_embedding = np.min(node_embeddings, axis=0)
  min_embedding = np.min(
      np.stack((min_node_embedding,
                min_edge_embedding)),
      axis=0)
  max_edge_embedding = np.max(edge_embeddings, axis=0)
  max_node_embedding = np.max(node_embeddings, axis=0)
  max_embedding = np.max(
      np.stack((max_node_embedding,
                max_edge_embedding)),
      axis=0)
  delta_embedding = max_embedding - min_embedding

  if not disable_pbar:
    log.info("Scaling nodes to 0-1 hypercube")
  with Pool(workers,
            initializer=_init_scale_alg_dist,
            initargs=(node_embeddings, #embedding
                      min_embedding, #min_embedding
                      delta_embedding #delta_embedding
                      )) as pool:
    with tqdm(total=len(hypergraph.node), disable=disable_pbar) as pbar:
      for idx, emb in pool.imap(_scale_alg_dist, hypergraph.node):
        node_embeddings[idx, :] = emb
        pbar.update(1)

  if not disable_pbar:
    log.info("Scaling edges to 0-1 hypercube")
  with Pool(workers,
            initializer=_init_scale_alg_dist,
            initargs=(edge_embeddings, #embedding
                      min_embedding, #min_embedding
                      delta_embedding #delta_embedding
                      )) as pool:
    with tqdm(total=len(hypergraph.edge), disable=disable_pbar) as pbar:
      for idx, emb in pool.imap(_scale_alg_dist, hypergraph.edge):
        edge_embeddings[idx, :] = emb
        pbar.update(1)
  return node_embeddings, edge_embeddings


def EmbedAlgebraicDistance(
    hypergraph,
    dimension,
    iterations=20,
    run_in_parallel=True,
    disable_pbar=False):
  workers = multiprocessing.cpu_count() if run_in_parallel else 1

  num_nodes = max(hypergraph.node) + 1
  num_edges = max(hypergraph.edge) + 1

  log.info("Random Initialization")
  # all embeddings are in 0-1 interval
  node_embeddings = np.random.random((num_nodes, dimension))
  edge_embeddings = np.random.random((num_edges, dimension))

  log.info("Getting node-edge matrix")
  node2edges = ToCsrMatrix(hypergraph)
  log.info("Getting edge-node matrix")
  edge2nodes = ToEdgeCsrMatrix(hypergraph)

  log.info("Performing iterations of Algebraic Distance Calculations")
  for iteration in tqdm(range(iterations), disable=disable_pbar):

    node_embeddings, edge_embeddings = _helper_update_embeddings(
        hypergraph,
        node_embeddings,
        edge_embeddings,
        node2edges,
        edge2nodes,
        workers,
        disable_pbar=True)
    node_embeddings, edge_embeddings = _helper_scale_embeddings(
        hypergraph,
        node_embeddings,
        edge_embeddings,
        workers,
        disable_pbar=True)

  embedding = HypergraphEmbedding()
  embedding.dim = dimension
  embedding.method_name = "AlgebraicDistance"
  for node_idx in hypergraph.node:
    embedding.node[node_idx].values.extend(node_embeddings[node_idx, :])
  for edge_idx in hypergraph.edge:
    embedding.edge[edge_idx].values.extend(edge_embeddings[edge_idx, :])
  return embedding
