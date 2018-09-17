# This module contains the new hypergraph2vec process.
# Its primary functions, PrecomputeSimilarities and GetModel
# Are used in the "embedding" module to actually perform this task

from .hypergraph_util import *
from . import HypergraphEmbedding
from .hypergraph2vec import SimilarityRecord
from .hypergraph2vec import _similarity_values_to_model_input

import numpy as np
import scipy as sp
from scipy.spatial.distance import jaccard
from scipy.sparse import csr_matrix

from random import random, sample, choice
import logging
import multiprocessing
from multiprocessing import Pool
from itertools import combinations, permutations, product
from time import time
from tqdm import tqdm
from collections import namedtuple

import keras
from keras.models import Model
from keras.layers import Input, Embedding, Multiply, Dense
from keras.layers import Dot, Flatten, Average, Activation
from keras.callbacks import TensorBoard

log = logging.getLogger()


def PrecomputeBinarySimilarities(
    hypergraph,
    num_neighbors,
    run_in_parallel=True):

  num_cores = multiprocessing.cpu_count() if run_in_parallel else 1
  # return value
  similarity_records = []

  log.info("Converting hypergraph to node-major sparse matrix")
  node2edge = ToCsrMatrix(hypergraph)
  log.info("Getting 1st order node neighbors")
  node2node_neighbors = node2edge * node2edge.T

  log.info("Sampling node-node probabilities")
  rows, cols = node2node_neighbors.nonzero()
  for row_idx, col_idx in tqdm(zip(rows, cols), total=len(rows)):
    similarity_records.append(
        SimilarityRecord(
            left_node_idx=row_idx,
            right_node_idx=col_idx,
            node_node_prob=1))

  log.info("Converting hypergraph to edge-major sparse matrix")
  edge2node = ToEdgeCsrMatrix(hypergraph)
  log.info("Getting 1st order edge neighbors")
  edge2edge_neighbors = edge2node * edge2node.T
  log.info("Sampling edge-edge probabilities")
  rows, cols = edge2edge_neighbors.nonzero()
  for row_idx, col_idx in tqdm(zip(rows, cols), total=len(rows)):
    similarity_records.append(
        SimilarityRecord(
            left_edge_idx=row_idx,
            right_edge_idx=col_idx,
            edge_edge_prob=1))

  log.info("Getting node-edge relationships")
  node2second_edge = node2node_neighbors * node2edge
  rows, cols = node2second_edge.nonzero()
  for node_idx, edge_idx in tqdm(zip(rows, cols), total=len(rows)):
    neighbor_edges = list(node2edge[node_idx, :].nonzero()[1])
    neighbor_edges = sample(
        neighbor_edges,
        min(num_neighbors,
            len(neighbor_edges)))
    neighbor_nodes = list(edge2node[edge_idx, :].nonzero()[1])
    neighbor_nodes = sample(
        neighbor_nodes,
        min(num_neighbors,
            len(neighbor_nodes)))
    similarity_records.append(
        SimilarityRecord(
            left_node_idx=node_idx,
            right_edge_idx=edge_idx,
            edges_containing_node=neighbor_edges,
            nodes_in_edge=neighbor_nodes,
            node_edge_prob=1))

  log.info("Getting edge-node relationships")
  edge2second_node = edge2edge_neighbors * edge2node
  rows, cols = edge2second_node.nonzero()
  for edge_idx, node_idx in tqdm(zip(rows, cols), total=len(rows)):
    neighbor_edges = list(node2edge[node_idx, :].nonzero()[1])
    neighbor_edges = sample(
        neighbor_edges,
        min(num_neighbors,
            len(neighbor_edges)))
    neighbor_nodes = list(edge2node[edge_idx, :].nonzero()[1])
    neighbor_nodes = sample(
        neighbor_nodes,
        min(num_neighbors,
            len(neighbor_nodes)))
    similarity_records.append(
        SimilarityRecord(
            left_node_idx=node_idx,
            right_edge_idx=edge_idx,
            edges_containing_node=neighbor_edges,
            nodes_in_edge=neighbor_nodes,
            node_edge_prob=1))

  return similarity_records


def GetModel(hypergraph, dimension, num_neighbors):
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
  node_node_prob = Activation(
      "sigmoid",
      name="node_node_prob")(
          Dot(1)([left_node_vec,
                  right_node_vec]))
  edge_edge_prob = Activation(
      "sigmoid",
      name="edge_edge_prob")(
          Dot(1)([left_edge_vec,
                  right_edge_vec]))

  # Get neighborhood embeddings
  sig = Activation("sigmoid")
  nodes_dot_sigs = [
      sig(Dot(1)([Flatten()(node_emb(node)),
                  left_node_vec])) for node in nodes_in_edge
  ]
  edges_dot_sigs = [
      sig(Dot(1)([Flatten()(edge_emb(edge)),
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
  model.compile(optimizer="adagrad", loss="kullback_leibler_divergence")
  return model


################################################################################
# Hooks for runner                                                             #
################################################################################


def EmbedHypergraphBinary(
    hypergraph,
    dimension,
    num_neighbors=10,
    batch_size=256,
    epochs=15):
  similarity_records = PrecomputeBinarySimilarities(hypergraph, num_neighbors)
  input_features, output_probs = _similarity_values_to_model_input(similarity_records, num_neighbors)
  model = GetModel(hypergraph, dimension, num_neighbors)

  tb_log = "/tmp/logs/{}".format(time())
  log.info("Follow along at %s", tb_log)
  tensorboard = TensorBoard(log_dir=tb_log)

  model.fit(
      input_features,
      output_probs,
      batch_size=batch_size,
      epochs=epochs,
      callbacks=[tensorboard])

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
