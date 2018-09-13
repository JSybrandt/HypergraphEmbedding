# This file contains embedding objects to project hypergraph nodes and/or edges
# into a dense vector space.

from . import HypergraphEmbedding
from .hypergraph_util import *
from .hypergraph2vec import *

import numpy as np
import scipy as sp
from scipy.spatial.distance import jaccard
from scipy.sparse import csr_matrix

from sklearn.decomposition import NMF
from collections.abc import Mapping
from random import random
import logging
from node2vec import Node2Vec
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

log = logging.getLogger()

global EMBEDDING_OPTIONS


def Embed(args, hypergraph):
  log.info("Checking embedding dimensionality is smaller than # nodes/edges")
  assert min(len(hypergraph.node),
             len(hypergraph.edge)) > args.embedding_dimension

  log.info(
      "Embedding using method %s with %i dim",
      args.embedding_method,
      args.embedding_dimension)
  embedding = EMBEDDING_OPTIONS[args.embedding_method](
      hypergraph,
      args.embedding_dimension)
  log.info(
      "Embedding contains %i node and %i edge vectors",
      len(embedding.node),
      len(embedding.edge))
  return embedding


def EmbedSvd(hypergraph, dimension):
  assert dimension < len(hypergraph.node)
  assert dimension < len(hypergraph.edge)
  assert dimension > 0

  embedding = HypergraphEmbedding()
  embedding.dim = dimension
  embedding.method_name = "SVD"

  matrix = ToCsrMatrix(hypergraph).asfptype()
  U, _, V = sp.sparse.linalg.svds(matrix, dimension)
  for node_idx in hypergraph.node:
    embedding.node[node_idx].values.extend(U[node_idx, :])
  for edge_idx in hypergraph.edge:
    embedding.edge[edge_idx].values.extend(V[:, edge_idx])

  return embedding


def EmbedRandom(hypergraph, dimension):
  assert dimension > 0

  embedding = HypergraphEmbedding()
  embedding.dim = dimension
  embedding.method_name = "Random"

  for node_idx in hypergraph.node:
    embedding.node[node_idx].values.extend([random() for _ in range(dimension)])
  for edge_idx in hypergraph.edge:
    embedding.edge[edge_idx].values.extend([random() for _ in range(dimension)])
  return embedding


def EmbedNMF(hypergraph, dimension):
  assert dimension > 0
  assert dimension < len(hypergraph.node)
  assert dimension < len(hypergraph.edge)

  embedding = HypergraphEmbedding()
  embedding.dim = dimension
  embedding.method_name = "NMF"

  matrix = ToCsrMatrix(hypergraph)
  nmf_model = NMF(dimension)
  W = nmf_model.fit_transform(matrix)
  H = nmf_model.components_
  for node_idx in hypergraph.node:
    embedding.node[node_idx].values.extend(W[node_idx, :])
  for edge_idx in hypergraph.edge:
    embedding.edge[edge_idx].values.extend(H[:, edge_idx])

  return embedding


def EmbedNode2VecBipartide(
    hypergraph,
    dimension,
    p=1,
    q=1,
    num_walks_per_node=10,
    walk_length=5,
    window=3,
    run_in_parallel=True):
  assert dimension > 0
  assert p >= 0 and p <= 1
  assert q >= 1 and q <= 1
  assert num_walks_per_node > 0
  assert walk_length > 0
  assert len(hypergraph.node) > 0
  assert len(hypergraph.edge) > 0

  workers = multiprocessing.cpu_count() if run_in_parallel else 1

  embedding = HypergraphEmbedding()
  embedding.dim = dimension
  embedding.method_name = "Node2VecBipartide({})".format(walk_length)

  bipartide = ToBipartideNxGraph(hypergraph)
  embedder = Node2Vec(
      bipartide,
      p=p,
      q=q,
      dimensions=dimension,
      walk_length=walk_length,
      num_walks=num_walks_per_node,
      workers=workers)
  model = embedder.fit(window=window, min_count=1, batch_words=4)

  max_node_idx = max(i for i, _ in hypergraph.node.items())

  for node_idx in hypergraph.node:
    assert str(node_idx) in model.wv
    embedding.node[node_idx].values.extend(model.wv[str(node_idx)])
  for edge_idx in hypergraph.edge:
    mod_edge_idx = max_node_idx + edge_idx + 1
    assert str(mod_edge_idx) in model.wv
    embedding.edge[edge_idx].values.extend(model.wv[str(mod_edge_idx)])

  return embedding


def EmbedNode2VecClique(
    hypergraph,
    dimension,
    p=1,
    q=1,
    num_walks_per_node=10,
    walk_length=5,
    window=3,
    run_in_parallel=True):
  assert dimension > 0
  assert p >= 0 and p <= 1
  assert q >= 1 and q <= 1
  assert num_walks_per_node > 0
  assert walk_length > 0
  assert len(hypergraph.node) > 0
  assert len(hypergraph.edge) > 0

  workers = multiprocessing.cpu_count() if run_in_parallel else 1

  embedding = HypergraphEmbedding()
  embedding.dim = dimension
  embedding.method_name = "Node2VecClique({})".format(walk_length)

  clique = ToCliqueNxGraph(hypergraph)
  embedder = Node2Vec(
      clique,
      p=p,
      q=q,
      dimensions=dimension,
      walk_length=walk_length,
      num_walks=num_walks_per_node,
      workers=workers)
  model = embedder.fit(window=window, min_count=1, batch_words=4)

  for node_idx in hypergraph.node:
    if str(node_idx) not in model.wv:
      # disconnected nodes (nodes occuring in 1 edge alone)
      # should be set to null
      embedding.node[node_idx].values.extend([0 for i in range(dimension)])
    else:
      embedding.node[node_idx].values.extend(model.wv[str(node_idx)])
  # Compute edge as centroid from nodes
  for edge_idx, edge in hypergraph.edge.items():
    edge_vec = np.mean(
        [embedding.node[node_idx].values for node_idx in edge.nodes],
        axis=0)
    embedding.edge[edge_idx].values.extend(edge_vec)

  return embedding


def EmbedHypergraph(
    hypergraph,
    dimension,
    num_neighbors=5,
    pos_samples=100,
    neg_samples=0,
    batch_size=256,
    epochs=5):
  input_features, output_probs = PrecomputeSimilarities(hypergraph,
                                                        num_neighbors,
                                                        pos_samples,
                                                        neg_samples)
  model = GetModel(hypergraph, dimension, num_neighbors)
  model.fit(input_features, output_probs, batch_size=batch_size, epochs=epochs)

  log.info("Recording Embeddings")

  node_weights = model.get_layer("node_embedding").get_weights()[0]
  edge_weights = model.get_layer("edge_embedding").get_weights()[0]

  embedding = HypergraphEmbedding()
  embedding.dim = dimension
  embedding.method_name = "Hypergraph"

  for node_idx in hypergraph.node:
    embedding.node[node_idx].values.extend(node_weights[node_idx + 1])
  for edge_idx in hypergraph.edge:
    embedding.edge[edge_idx].values.extend(edge_weights[edge_idx + 1])
  return embedding


def EmbedHypergraphPlusPlus(
    hypergraph,
    dimension,
    num_neighbors=10,
    num_walks_per_node=50,
    max_walk_length=7,
    walk_tolerance=0.001,
    batch_size=256,
    epochs=5):
  input_features, output_probs = PrecomputeSimilaritiesPlusPlus(
      hypergraph,
      num_neighbors,
      num_walks_per_node,
      max_walk_length,
      walk_tolerance)
  model = GetModel(hypergraph, dimension, num_neighbors)
  model.fit(input_features, output_probs, batch_size=batch_size, epochs=epochs)

  log.info("Recording Embeddings")

  node_weights = model.get_layer("node_embedding").get_weights()[0]
  edge_weights = model.get_layer("edge_embedding").get_weights()[0]

  embedding = HypergraphEmbedding()
  embedding.dim = dimension
  embedding.method_name = "Hypergraph++"

  for node_idx in hypergraph.node:
    embedding.node[node_idx].values.extend(node_weights[node_idx + 1])
  for edge_idx in hypergraph.edge:
    embedding.edge[edge_idx].values.extend(edge_weights[edge_idx + 1])
  return embedding


################################################################################
# AlgebraicDistance - Helper and runner                                        #
################################################################################

_shared_info = {}

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

  b_emb = sum(B2emb[b_idx, :] for b_idx in A2B[a_idx, :].nonzero()[1])
  b_emb /= A2B[a_idx, :].nnz

  return a_idx, (a_emb + b_emb) / 2


def _helper_update_embeddings(
    hypergraph,
    node_embeddings,
    edge_embeddings,
    node2edges,
    edge2nodes,
    workers,
    disable_pbar):
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

  for iteration in range(iterations):
    log.info("Iteration %i/%i", iteration, iterations)

    node_embeddings, edge_embeddings = _helper_update_embeddings(
        hypergraph,
        node_embeddings,
        edge_embeddings,
        node2edges,
        edge2nodes,
        workers,
        disable_pbar)
    node_embeddings, edge_embeddings = _helper_scale_embeddings(
        hypergraph,
        node_embeddings,
        edge_embeddings,
        workers,
        disable_pbar)

  embedding = HypergraphEmbedding()
  embedding.dim = dimension
  embedding.method_name = "AlgebraicDistance"
  for node_idx in hypergraph.node:
    embedding.node[node_idx].values.extend(node_embeddings[node_idx, :])
  for edge_idx in hypergraph.edge:
    embedding.edge[edge_idx].values.extend(edge_embeddings[edge_idx, :])
  return embedding


EMBEDDING_OPTIONS = {
    "SVD": EmbedSvd,
    "RANDOM": EmbedRandom,
    "NMF": EmbedNMF,
    "N2V3_BIPARTIDE": lambda h, d:EmbedNode2VecBipartide(h, d, walk_length=3),
    "N2V3_CLIQUE": lambda h, d: EmbedNode2VecClique(h, d, walk_length=3),
    "N2V5_BIPARTIDE": EmbedNode2VecBipartide,
    "N2V5_CLIQUE": EmbedNode2VecClique,
    "N2V7_BIPARTIDE": lambda h, d:EmbedNode2VecBipartide(h, d, walk_length=7),
    "N2V7_CLIQUE": lambda h, d: EmbedNode2VecClique(h, d, walk_length=7),
    "HYPERGRAPH": EmbedHypergraph,
    "HYPERGRAPH++": EmbedHypergraphPlusPlus,
    "ALG_DIST": EmbedAlgebraicDistance
}
