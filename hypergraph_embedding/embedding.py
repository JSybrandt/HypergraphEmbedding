# This file contains embedding objects to project hypergraph nodes and/or edges
# into a dense vector space.

from . import ToCsrMatrix
from . import HypergraphEmbedding
from .hypergraph_util import *
import scipy as sp
from scipy.spatial.distance import jaccard
import numpy as np
from sklearn.decomposition import NMF
from collections.abc import Mapping
from random import random, sample
import logging
from node2vec import Node2Vec
import multiprocessing
from itertools import combinations, permutations, product

import keras
from keras.layers import Input, Embedding, Multiply, Dense, Dot, Reshape, Add, Subtract, Concatenate, Flatten, Lambda, Average
from keras.models import Model
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

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

  matrix = ToCsrMatrix(hypergraph)
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
  embedding.method_name = "Node2VecBipartide"

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
  embedding.method_name = "Node2VecClique"

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


# Here on out is for the new method, and its helper functions


def _GetSecondDegConnections(mat):
  """
  Helper function. Given a csr_matrix, compute all neighboring rows.
  Result is an N x N matrix where if res[i,j] = 1 then row i and
  row j share a non-zero element
  """
  num = mat.shape[0]
  neighbors = sp.sparse.csr_matrix((num, num), dtype=np.bool)
  for idx_i, idx_j in combinations(range(num), 2):
    # If i and j share an edge
    if mat[idx_i, :].dot(mat[idx_j, :].T)[0, 0] > 0:
      neighbors[idx_i, idx_j] = 1
      neighbors[idx_j, idx_i] = 1
  # add self edge too
  for idx_i in range(num):
    neighbors[idx_i, idx_i] = 1
  return neighbors


def _GetNodeNeighbors(hypergraph, node2edges=None):
  """
    Returns a Csr matrix where if i,j=1 then node i and node j share an edge.
    If node2edges is set, then we skip the ToCsrMatrix computation on the
    hypergraph.
  """
  if node2edges is None:
    node2edges = ToCsrMatrix(hypergraph)
  return _GetSecondDegConnections(node2edges)


def _GetEdgeNeighbors(hypergraph, edge2nodes=None):
  """
    Returns a Csr matrix where if i,j=1 then edge i and edge j share a node.
    If edge2nodes is set, then we skip the ToCsrMatrix computation
  """
  if edge2nodes is None:
    edge2nodes = ToCscMatrix(hypergraph).T
  return _GetSecondDegConnections(edge2nodes)


def _PrecomputeSimilarities(hypergraph, num_neighbors):
  log.info("Precomputing similarities")
  node2edges = ToCsrMatrix(hypergraph)
  edge2nodes = ToCscMatrix(hypergraph).T

  node2neighbors = _GetNodeNeighbors(hypergraph, node2edges)
  edge2neighbors = _GetEdgeNeighbors(hypergraph, edge2nodes)

  def NodeNodeSim(idx_i, idx_j):
    edge_set_i = node2edges[idx_i, :]
    edge_set_j = node2edges[idx_j, :]
    return jaccard(edge_set_i, edge_set_j)

  def EdgeEdgeSim(idx_i, idx_j):
    node_set_i = edge2nodes[idx_i, :]
    node_set_j = edge2nodes[idx_j, :]
    return jaccard(node_set_i, node_set_j)

  def NodeEdgeSim(node_idx, edge_idx):
    edges_containing_node = node2edges[node_idx, :]
    nodes_in_edge = edge2nodes[edge_idx, :]
    return jaccard(node2neighbors[node_idx, :], nodes_in_edge) \
         * jaccard(edge2neighbors[edge_idx, :], edges_containing_node)

  def sample_column_indices(idx, matrix):
    cols = list(matrix[idx, :].nonzero()[1])
    if len(cols) > num_neighbors:
      cols = sample(cols, num_neighbors)
    return [c + 1 for c in cols]

  # precomputed array size
  rows = len(hypergraph.node) * (len(hypergraph.node)-1) \
       + len(hypergraph.edge) * (len(hypergraph.edge)-1) \
       + len(hypergraph.node) * len(hypergraph.edge)
  # idx, bool(types) idx of neighbors

  log.info("Instantiating arrays")
  left_node_idx = np.zeros((rows,), dtype=np.int32)
  left_edge_idx = np.zeros((rows,), dtype=np.int32)

  right_node_idx = np.zeros((rows,), dtype=np.int32)
  right_edge_idx = np.zeros((rows,), dtype=np.int32)

  edges_containing_node = [
      np.zeros((rows,
               ),
               dtype=np.int32) for _ in range(num_neighbors)
  ]
  nodes_in_edge = [
      np.zeros((rows,
               ),
               dtype=np.int32) for _ in range(num_neighbors)
  ]

  node_node_prob = np.zeros((rows,), dtype=np.float16)
  edge_edge_prob = np.zeros((rows,), dtype=np.float16)
  node_edge_prob = np.zeros((rows,), dtype=np.float16)

  row_idx = 0

  # Note, we are going to store indices + 1 because 0 is a mask value

  log.info("Computing all node-node probabilities")
  for i, j in permutations(hypergraph.node, 2):
    left_node_idx[row_idx] = i + 1
    right_node_idx[row_idx] = j + 1

    node_node_prob[row_idx] = NodeNodeSim(i, j)
    row_idx += 1

  log.info("Computing all edge-edge probabilities")
  for i, j in permutations(hypergraph.edge, 2):
    left_edge_idx[row_idx] = i + 1
    right_edge_idx[row_idx] = j + 1

    edge_edge_prob[row_idx] = EdgeEdgeSim(i, j)
    row_idx += 1

  log.info("Computing all node-edge probabilities")
  for n, e in product(hypergraph.node, hypergraph.edge):
    left_node_idx[row_idx] = n + 1
    right_edge_idx[row_idx] = e + 1

    for col, neigh_sample in enumerate(sample_column_indices(n, node2edges)):
      edges_containing_node[col][row_idx] = neigh_sample
    for col, neigh_sample in enumerate(sample_column_indices(e, edge2nodes)):
      nodes_in_edge[col][row_idx] = neigh_sample

    node_edge_prob[row_idx] = NodeEdgeSim(n, e)
    row_idx += 1

  return ([left_node_idx,
           left_edge_idx,
           right_node_idx,
           right_edge_idx] + edges_containing_node + nodes_in_edge,
          [node_node_prob,
           edge_edge_prob,
           node_edge_prob])


def _GetModel(hypergraph, dimension, num_neighbors):
  log.info("Constructing Keras Model")

  max_node_idx = max([i for i in hypergraph.node])
  max_edge_idx = max([i for i in hypergraph.edge])

  left_node_idx = Input((1,), name="left_node_idx", dtype=np.int32)
  left_edge_idx = Input((1,), name="left_edge_idx", dtype=np.int32)
  right_node_idx = Input((1,), name="right_node_idx", dtype=np.int32)
  right_edge_idx = Input((1,), name="right_edge_idx", dtype=np.int32)
  edges_containing_node = [
      Input((1,
            ),
            dtype=np.int32,
            name="edges_containing_node_{}".format(i))
      for i in range(num_neighbors)
  ]
  nodes_in_edge = [
      Input((1,
            ),
            dtype=np.int32,
            name="nodes_in_edge_{}".format(i)) for i in range(num_neighbors)
  ]

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

  # Pad each input, get embeddings
  left_node_vec = Flatten()(node_emb(left_node_idx))
  left_edge_vec = Flatten()(edge_emb(left_edge_idx))
  right_node_vec = Flatten()(node_emb(right_node_idx))
  right_edge_vec = Flatten()(edge_emb(right_edge_idx))

  # calculate expected probabilities
  node_node_prob = Dense(
      1,
      activation="sigmoid",
      name="node_node_prob")(
          Dot(0)([left_node_vec,
                  right_node_vec]))
  edge_edge_prob = Dense(
      1,
      activation="sigmoid",
      name="edge_edge_prob")(
          Dot(0)([left_edge_vec,
                  right_edge_vec]))

  # Get neighborhood embeddings
  nodes_dot_sigs = [
      Dense(1,
            activation="sigmoid")(
                Dot(0)([Flatten()(node_emb(node)),
                        left_node_vec])) for node in nodes_in_edge
  ]
  edges_dot_sigs = [
      Dense(1,
            activation="sigmoid")(
                Dot(0)([Flatten()(edge_emb(edge)),
                        right_edge_vec])) for edge in edges_containing_node
  ]

  node_sig_avg = Average()(nodes_dot_sigs)
  edge_sig_avg = Average()(edges_dot_sigs)
  node_edge_prob = Multiply(name="node_edge_prob")([node_sig_avg, edge_sig_avg])
  model = Model(
      inputs=[left_node_idx,
              left_edge_idx,
              right_node_idx,
              right_edge_idx] + edges_containing_node + nodes_in_edge,
      outputs=[node_node_prob,
               edge_edge_prob,
               node_edge_prob])

  model.compile(optimizer="rmsprop", loss="kullback_leibler_divergence")
  return model


def EmbedHypergraph(hypergraph, dimension, num_neighbors=5):
  input_features, output_probs = _PrecomputeSimilarities(hypergraph, num_neighbors)
  model = _GetModel(hypergraph, dimension, num_neighbors)
  model.fit(input_features, output_probs, batch_size=1)

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


EMBEDDING_OPTIONS = {
    "SVD": EmbedSvd,
    "RANDOM": EmbedRandom,
    "NMF": EmbedNMF,
    "N2V_BIPARTIDE": EmbedNode2VecBipartide,
    "N2V_CLIQUE": EmbedNode2VecClique,
    "HYPERGRAPH": EmbedHypergraph
}
