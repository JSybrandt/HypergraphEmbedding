# This file contains embedding objects to project hypergraph nodes and/or edges
# into a dense vector space.

from . import HypergraphEmbedding
from .hypergraph_util import *
from .algebraic_distance import EmbedAlgebraicDistance
from .auto_encoder import EmbedAutoEncoder

from .hg2v_model import *
from .hg2v_sample import *
from .hg2v_weighting import *

import numpy as np
import scipy as sp
from scipy.spatial.distance import jaccard
from scipy.sparse import csr_matrix

from sklearn.decomposition import NMF
from collections.abc import Mapping
from pathlib import Path
from random import random
import logging
from node2vec import Node2Vec
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from time import time

from keras.callbacks import EarlyStopping

log = logging.getLogger()

global EMBEDDING_OPTIONS
global DEBUG_SUMMARY_OPTIONS


def Embed(args, hypergraph):
  log.info("Checking embedding dimensionality is smaller than # nodes/edges")
  assert min(len(hypergraph.node), len(
      hypergraph.edge)) > args.embedding_dimension
  log.info("Embedding using method %s with %i dim", args.embedding_method,
           args.embedding_dimension)
  if args.embedding_debug_summary:
    debug_summary_path = Path(args.embedding_debug_summary)
    log.info("... and writing summary to %s", debug_summary_path)
    embedding = EMBEDDING_OPTIONS[args.embedding_method](
        hypergraph,
        args.embedding_dimension,
        debug_summary_path=debug_summary_path)
  else:
    embedding = EMBEDDING_OPTIONS[args.embedding_method](
        hypergraph, args.embedding_dimension)
  log.info("Embedding contains %i node and %i edge vectors",
           len(embedding.node), len(embedding.edge))
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


def EmbedNode2VecBipartide(hypergraph,
                           dimension,
                           p=1,
                           q=1,
                           num_walks_per_node=10,
                           walk_length=5,
                           window=3,
                           run_in_parallel=True,
                           disable_pbar=False):
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
      workers=workers,
      quiet=disable_pbar)
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


def EmbedNode2VecClique(hypergraph,
                        dimension,
                        p=1,
                        q=1,
                        num_walks_per_node=10,
                        walk_length=5,
                        window=3,
                        run_in_parallel=True,
                        disable_pbar=False):
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
      workers=workers,
      quiet=disable_pbar)
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
        [embedding.node[node_idx].values for node_idx in edge.nodes], axis=0)
    embedding.edge[edge_idx].values.extend(edge_vec)

  return embedding


################################################################################
# Hypergraph2Vec Combinations                                                  #
################################################################################


def _hypergraph2vec_skeleton(hypergraph, dimension, sampler_fn,
                             samples_to_input_fn, model_fn, fit_batch_size,
                             fit_epochs, visualization_fn, disable_pbar):
  log.info("Sampling")
  samples = sampler_fn(hypergraph)
  if visualization_fn is not None:
    try:
      visualization_fn(samples)
    except:
      log.error("Something happened when visualizing...")
  log.info("Converting samples to model input")
  input_features, output_probs = samples_to_input_fn(samples)
  log.info("Getting model")
  model = model_fn(hypergraph)

  tb_log = "/tmp/logs/{}".format(time())
  log.info("Follow along at %s", tb_log)
  tensorboard = TensorBoard(log_dir=tb_log)

  stopper = EarlyStopping(monitor="loss")

  model.fit(
      input_features,
      output_probs,
      batch_size=fit_batch_size,
      epochs=fit_epochs,
      callbacks=[tensorboard, stopper],
      verbose=0 if disable_pbar else 1)

  log.info("Recording embeddings.")
  return KerasModelToEmbedding(hypergraph, model)


def EmbedHg2vBoolean(hypergraph,
                     dimension,
                     num_neighbors=5,
                     num_samples=200,
                     batch_size=256,
                     epochs=10,
                     debug_summary_path=None,
                     disable_pbar=False):
  sampler_fn = lambda hg: BooleanSamples(hg,
                                         num_neighbors=num_neighbors,
                                         num_samples=num_samples,
                                         disable_pbar=disable_pbar)
  samples_to_input_fn = lambda samples: SamplesToModelInput(
      samples,
      num_neighbors=num_neighbors,
      weighted=False)
  if debug_summary_path:
    vis_fn = lambda samples: PlotDistributions(debug_summary_path, samples)
  else:
    vis_fn = None
  model_fn = lambda hg: BooleanModel(hg,
                                     dimension=dimension,
                                     num_neighbors=num_neighbors)
  embedding = _hypergraph2vec_skeleton(hypergraph, dimension, sampler_fn,
                                       samples_to_input_fn, model_fn,
                                       batch_size, epochs, vis_fn, disable_pbar)
  embedding.method_name = "Hypergraph2Vec Boolean"
  return embedding


def EmbedHg2vAdjJaccard(hypergraph,
                        dimension,
                        num_neighbors=5,
                        num_samples=200,
                        batch_size=256,
                        epochs=10,
                        debug_summary_path=None,
                        disable_pbar=False):
  node2weight, edge2weight = UniformWeight(hypergraph)
  sampler_fn = lambda hg: WeightedJaccardSamples(hg,
                                                 node2weight,
                                                 edge2weight,
                                                 num_neighbors=num_neighbors,
                                                 num_samples=num_samples,
                                                 disable_pbar=disable_pbar)
  samples_to_input_fn = lambda samples: SamplesToModelInput(
      samples,
      num_neighbors=num_neighbors,
      weighted=False)
  if debug_summary_path:
    vis_fn = lambda samples: PlotDistributions(debug_summary_path, samples)
  else:
    vis_fn = None
  model_fn = lambda hg: UnweightedFloatModel(hg,
                                             dimension=dimension,
                                             num_neighbors=num_neighbors)
  embedding = _hypergraph2vec_skeleton(hypergraph, dimension, sampler_fn,
                                       samples_to_input_fn, model_fn,
                                       batch_size, epochs, vis_fn, disable_pbar)
  embedding.method_name = "Hypergraph2Vec AdjJaccard"
  return embedding


def EmbedHg2vNeighborhoodWeightedJaccard(hypergraph,
                                         dimension,
                                         alpha=0,
                                         num_neighbors=5,
                                         num_samples=200,
                                         batch_size=256,
                                         epochs=10,
                                         debug_summary_path=None,
                                         disable_pbar=False):
  node2feature, edge2feature = WeightByNeighborhood(hypergraph, alpha)
  sampler_fn = lambda hg: WeightedJaccardSamples(hg,
                                                  node2feature,
                                                 edge2feature,
                                                 num_neighbors=num_neighbors,
                                                 num_samples=num_samples,
                                                 disable_pbar=disable_pbar)
  samples_to_input_fn = lambda samples: SamplesToModelInput(
      samples,
      num_neighbors=num_neighbors,
      weighted=False)
  if debug_summary_path:
    vis_fn = lambda samples: PlotDistributions(debug_summary_path, samples)
  else:
    vis_fn = None
  model_fn = lambda hg: UnweightedFloatModel(hg,
                                             dimension=dimension,
                                             num_neighbors=num_neighbors)
  embedding = _hypergraph2vec_skeleton(hypergraph, dimension, sampler_fn,
                                       samples_to_input_fn, model_fn,
                                       batch_size, epochs, vis_fn, disable_pbar)
  embedding.method_name = "Hypergraph2Vec Neighborhood Weighted Jaccard"
  return embedding


def EmbedHg2vSpanWeightedJaccard(hypergraph,
                                 dimension,
                                 alpha=0,
                                 num_neighbors=5,
                                 num_samples=200,
                                 batch_size=256,
                                 epochs=10,
                                 debug_summary_path=None,
                                 disable_pbar=False):
  node2weight, edge2weight = WeightByAlgebraicSpan(hypergraph, alpha)
  sampler_fn = lambda hg: WeightedJaccardSamples(hg,
                                                 node2weight,
                                                 edge2weight,
                                                 num_neighbors=num_neighbors,
                                                 num_samples=num_samples,
                                                 disable_pbar=disable_pbar)
  samples_to_input_fn = lambda samples: SamplesToModelInput(
      samples,
      num_neighbors=num_neighbors,
      weighted=False)
  if debug_summary_path:
    vis_fn = lambda samples: PlotDistributions(debug_summary_path, samples)
  else:
    vis_fn = None
  model_fn = lambda hg: UnweightedFloatModel(hg,
                                             dimension=dimension,
                                             num_neighbors=num_neighbors)
  embedding = _hypergraph2vec_skeleton(hypergraph, dimension, sampler_fn,
                                       samples_to_input_fn, model_fn,
                                       batch_size, epochs, vis_fn, disable_pbar)
  embedding.method_name = "Hypergraph2Vec Span Weighted Jaccard"
  return embedding


def EmbedHg2vDistWeightedJaccard(hypergraph,
                                 dimension,
                                 alpha=0,
                                 num_neighbors=5,
                                 num_samples=200,
                                 batch_size=256,
                                 epochs=10,
                                 debug_summary_path=None,
                                 disable_pbar=False):

  log.info("Embedding weighted by algebraic distance.")
  alg_emb = EmbedAlgebraicDistance(hypergraph, dimension=10, iterations=20)
  euc_norm = lambda x: np.linalg.norm(x, ord=2)
  node2weight, edge2weight = WeightByDistance(hypergraph, alpha, alg_emb,
                                              euc_norm, disable_pbar)
  sampler_fn = lambda hg: WeightedJaccardSamples(hg,
                                                 node2weight,
                                                 edge2weight,
                                                 num_neighbors=num_neighbors,
                                                 num_samples=num_samples,
                                                 disable_pbar=disable_pbar)
  samples_to_input_fn = lambda samples: SamplesToModelInput(
      samples,
      num_neighbors=num_neighbors,
      weighted=False)
  if debug_summary_path:
    vis_fn = lambda samples: PlotDistributions(debug_summary_path, samples)
  else:
    vis_fn = None
  model_fn = lambda hg: UnweightedFloatModel(hg,
                                             dimension=dimension,
                                             num_neighbors=num_neighbors)
  embedding = _hypergraph2vec_skeleton(hypergraph, dimension, sampler_fn,
                                       samples_to_input_fn, model_fn,
                                       batch_size, epochs, vis_fn, disable_pbar)
  embedding.method_name = "Hypergraph2Vec Dist Weighted Jaccard"
  return embedding


def EmbedHg2vAlgDist(hypergraph,
                     dimension,
                     alpha=0,
                     num_neighbors=5,
                     num_samples=200,
                     batch_size=256,
                     epochs=10,
                     debug_summary_path=None,
                     disable_pbar=False):

  log.info("Embedding weighted by algebraic distance.")
  alg_emb = EmbedAlgebraicDistance(hypergraph, dimension=10, iterations=20)
  euc_norm = lambda x: np.linalg.norm(x, ord=2)
  node2edge_weight, edge2node_weight = WeightByDistance(
      hypergraph, alpha, alg_emb, euc_norm, disable_pbar)
  node2node_weight, edge2edge_weight = WeightBySameTypeDistance(
      hypergraph, alpha, alg_emb, euc_norm, disable_pbar=disable_pbar)
  sampler_fn = lambda hg: AlgebraicDistanceSamples(hg,
                                                   node2edge_weight,
                                                   edge2node_weight,
                                                   node2node_weight,
                                                   edge2edge_weight,
                                                   num_neighbors=num_neighbors,
                                                   num_samples=num_samples,
                                                   disable_pbar=disable_pbar)
  samples_to_input_fn = lambda samples: SamplesToModelInput(
      samples,
      num_neighbors=num_neighbors,
      weighted=False)
  if debug_summary_path:
    vis_fn = lambda samples: PlotDistributions(debug_summary_path, samples)
  else:
    vis_fn = None
  model_fn = lambda hg: UnweightedFloatModel(hg,
                                             dimension=dimension,
                                             num_neighbors=num_neighbors)
  embedding = _hypergraph2vec_skeleton(hypergraph, dimension, sampler_fn,
                                       samples_to_input_fn, model_fn,
                                       batch_size, epochs, vis_fn, disable_pbar)
  embedding.method_name = "Hypergraph2Vec Dist Weighted Jaccard"
  return embedding


EMBEDDING_OPTIONS = {
    "SVD": EmbedSvd,
    "RANDOM": EmbedRandom,
    "NMF": EmbedNMF,
    "AUTO_ENCODER": EmbedAutoEncoder,
    "N2V3_BIPARTIDE": lambda h, d: EmbedNode2VecBipartide(h, d, walk_length=3),
    "N2V3_CLIQUE": lambda h, d: EmbedNode2VecClique(h, d, walk_length=3),
    "N2V5_BIPARTIDE": EmbedNode2VecBipartide,
    "N2V5_CLIQUE": EmbedNode2VecClique,
    "N2V7_BIPARTIDE": lambda h, d: EmbedNode2VecBipartide(h, d, walk_length=7),
    "N2V7_CLIQUE": lambda h, d: EmbedNode2VecClique(h, d, walk_length=7),
    "ALG_DIST": EmbedAlgebraicDistance,
    "HG2V_BOOLEAN": EmbedHg2vBoolean,
    "HG2V_ADJ_JAC": EmbedHg2vAdjJaccard,
    "HG2V_NEIGH_JAC": EmbedHg2vNeighborhoodWeightedJaccard,
    "HG2V_SPAN_JAC": EmbedHg2vSpanWeightedJaccard,
    # "HG2V_ALG_DIST": EmbedHg2vDistWeightedJaccard,
    "HG2V_ALG_DIST": EmbedHg2vAlgDist,
}

# Only include here if the embedding function supports the keyword argument
# debug_summary_path
DEBUG_SUMMARY_OPTIONS = {
    "HG2V_BOOLEAN",
    "HG2V_ADJ_JAC",
    "HG2V_NEIGH_JAC",
    "HG2V_SPAN_JAC",
    "HG2V_ALG_DIST",
}
