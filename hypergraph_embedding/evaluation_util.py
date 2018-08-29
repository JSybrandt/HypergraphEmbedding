# This file provides evaluation utilities for hypergraph experiments

from random import random
from . import Hypergraph, EvaluationMetrics
from .hypergraph_util import *
from scipy.spatial.distance import cosine
import numpy as np
from collections import namedtuple
import logging
import multiprocessing
from multiprocessing import Pool, Manager
from itertools import product

log = logging.getLogger()


def RunLinkPrediction(
    hypergraph,
    embedding,
    removal_probability,
    eval_only_removed=True):
  log.info(
      "Removing links from original hypergraph with prob %f",
      removal_probability)
  new_graph, removed_links = RemoveRandomConnections(hypergraph, removal_probability)
  log.info("Predicting links on subset graph")
  predicted_links = CommunityPrediction(
      new_graph,
      embedding,
      missing_links=removed_links if eval_only_removed else None)
  log.info("Evaluting link prediction performance")
  metrics = CalculateCommunityPredictionMetrics(predicted_links, removed_links)
  if hypergraph.HasField("name"):
    metrics.hypergraph_name = hypergraph.name
  if embedding.HasField("method_name"):
    metrics.embedding_method = embedding.method_name
  metrics.embedding_dim = embedding.dim
  return metrics


def RemoveRandomConnections(original_hypergraph, probability):
  """
    This function creates a new hypergraph where each node-edge with randomly
    sampled nodes removed from communities.
    input:
      - hypergraph proto object
      - number between 0 and 1
    outut:
      - new hypergraph with removed edges
      - list of all removed node-edge pairs
    """
  assert probability >= 0
  assert probability <= 1

  # stores (node_idx, edge_idx) tuples
  removed_connections = []

  new_hg = Hypergraph()
  for node_idx, node in original_hypergraph.node.items():
    for edge_idx in node.edges:
      edge = original_hypergraph.edge[edge_idx]

      # If we are going to drop this connection
      if random() < probability and probability > 0:
        removed_connections.append((node_idx, edge_idx))
      else:
        AddNodeToEdge(
            new_hg,
            node_idx,
            edge_idx,
            node.name if node.HasField("name") else None,
            edge.name if edge.HasField("name") else None)
  return new_hg, removed_connections


_shared_info = {}


def _init_to_numpy(_embedding):
  _shared_info['embedding'] = _embedding


def _to_numpy(node_idx):
  return (
      node_idx,
      np.asarray(
          _shared_info['embedding'].node[node_idx].values,
          dtype=np.float32))


def _init_get_edge_centroid_range(
    _node2embedding,
    _hypergraph,
    _distance_function):
  _shared_info['node2embedding'] = _node2embedding
  _shared_info['hypergraph'] = _hypergraph
  _shared_info['distance_function'] = _distance_function


def _get_edge_centroid_range(edge_idx):
  points = [
      _shared_info['node2embedding'][i]
      for i in _shared_info['hypergraph'].edge[edge_idx].nodes
  ]
  centroid = np.mean(points, axis=0)
  max_dist = max(
      [_shared_info['distance_function'](centroid,
                                         vec) for vec in points])
  return (edge_idx, centroid, max_dist)


def _init_get_missing_links(
    _node2embedding,
    _edge2centroid,
    _edge2range,
    _hypergraph,
    _distance_function):
  _shared_info['node2embedding'] = _node2embedding
  _shared_info['edge2centroid'] = _edge2centroid
  _shared_info['edge2range'] = _edge2range
  _shared_info['hypergraph'] = _hypergraph
  _shared_info['distance_function'] = _distance_function


def _get_missing_links(indices):
  "Checks if this node is in any of the edges"
  node_idx, edge_idx = indices
  if edge_idx in _shared_info['hypergraph'].node[node_idx].edges:
    return None
  if node_idx not in _shared_info['node2embedding']:
    return None
  if edge_idx not in _shared_info['edge2centroid']:
    return None

  vec = _shared_info['node2embedding'][node_idx]
  centroid = _shared_info['edge2centroid'][edge_idx]
  if _shared_info['distance_function'](
      vec,
      centroid) <= _shared_info['edge2range'][edge_idx]:
    return (node_idx, edge_idx)
  return None


def CommunityPrediction(
    hypergraph,
    embedding,
    distance_function=cosine,
    missing_links=None,
    run_in_parallel=True):
  """
  Given a hypergraph (assumed to contain missing links) and a corresponding
  embedding, idetify missing node-edge connections. Performs this task by
  comparing cosine similarities of potential connections with existing ones.
  Note, drops edges with < 2 nodes.
  Note, all nodes must be embedded.
  input:
    - hypergraph: Hypergraph proto
    - embedding: Hypergraph proto
    - distance_function: mesure of distance between embeddings
    - missing_links: if set, only run predictions on specified pairs
  output:
    - list of missing node-edge pairs
  """

  assert embedding.dim > 0

  num_cores = multiprocessing.cpu_count() if run_in_parallel else 1

  log.info("Converting all node embeddings to numpy")
  with Pool(num_cores,
            initializer=_init_to_numpy,
            initargs=(embedding,
                     )) as pool:
    node2embedding = {x[0]: x[1] for x in pool.map(_to_numpy, embedding.node)}

  log.info("Finding edges with at least two nodes")
  edges = [idx for idx, edge in hypergraph.edge.items() if len(edge.nodes) >= 2]

  log.info("Identifying each edge's centroid and range")

  with Pool(num_cores,
            initializer=_init_get_edge_centroid_range,
            initargs=(node2embedding,
                      hypergraph,
                      distance_function)) as pool:
    edge_centroid_range = pool.map(_get_edge_centroid_range, edges)

  log.info("Indexing edges")
  edge2centroid = {x[0]: x[1] for x in edge_centroid_range}
  edge2range = {x[0]: x[2] for x in edge_centroid_range}

  predicted_links = []
  if not missing_links:
    log.info("Identifying missing links")
    missing_links = product(embedding.node, edges)

  log.info("Calculating whether each point is in each edge's range")
  with Pool(num_cores,
            initializer=_init_get_missing_links,
            initargs=(node2embedding,
                      edge2centroid,
                      edge2range,
                      hypergraph,
                      distance_function)) as pool:
    for res in pool.imap(_get_missing_links, missing_links, chunksize=250):
      if res:
        predicted_links.append(res)

  return predicted_links


def CalculateCommunityPredictionMetrics(
    predicted_connections,
    expected_connections):
  """
    Treats expected_connections as positive examples, and computes a Metrics
    named tuple.
    input:
      - predicted_connections: iterable of (node_idx, edge_idx) pairs
      - expected_connections: iterable of (node_idx, edge_idx) pairs
    output:
      - EvaluationMetrics proto containing all fields but accuracy
        and num_true_neg
    """
  prediction_set = set(predicted_connections)
  expected_set = set(expected_connections)

  intersection = prediction_set.intersection(expected_set)

  metrics = EvaluationMetrics()
  if len(prediction_set):
    metrics.precision = len(intersection) / len(prediction_set)
  if len(expected_set):
    metrics.recall = len(intersection) / len(expected_set)
  if metrics.precision + metrics.recall:
    metrics.f1 = 2 * metrics.precision * metrics.recall / (
        metrics.precision + metrics.recall)
  metrics.num_true_pos = len(intersection)
  metrics.num_false_pos = len(prediction_set) - len(intersection)
  metrics.num_false_neg = len(expected_set) - len(intersection)
  # num_true_neg poorly defined
  # accuracy poorly defined
  return metrics
