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
    num_neighbors=10,
    pos_samples=200,
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
    "HYPERGRAPH": EmbedHypergraph
}
