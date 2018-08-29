# This file provides evaluation utilities for hypergraph experiments

from random import random, choice
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


def RunLinkPrediction(hypergraph, embedding, removal_probability):
  log.info(
      "Removing links from original hypergraph with prob %f",
      removal_probability)
  new_graph, removed_links = RemoveRandomConnections(hypergraph, removal_probability)
  log.info("Removed %i links", len(removed_links))

  log.info("Sampling missing links for evaluation")
  bad_links = SampleMissingConnections(hypergraph, len(removed_links))
  log.info("Sampled %i links", len(bad_links))

  log.info("Predicting links on subset graph")
  predicted_links = CommunityPrediction(
      new_graph,
      embedding,
      bad_links + removed_links)

  log.info("Evaluting link prediction performance")
  metrics = CalculateCommunityPredictionMetrics(
      predicted_links,
      removed_links,
      bad_links)

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


def SampleMissingConnections(hypergraph, num_samples):
  """
    Given a hypergraph, Sample for random not-present node-edge connections.
    Used as negative training data for the link prediction task.
    input:
      - hypergraph: hypergraph proto
      - num_samples
    output:
      - list of node_idx, edge_idx pairs where node is not in edge
  """
  samples = set()
  num_nodes = len(hypergraph.node)
  num_edges = len(hypergraph.edge)
  assert num_samples < num_nodes * num_edges
  assert len(hypergraph.edge) > 0
  assert len(hypergraph.node) > 0

  nodes = [n for n in hypergraph.node]
  edges = [e for e in hypergraph.edge]

  num_tries_till_failure = 10 * num_samples
  while len(samples) < num_samples and num_tries_till_failure:
    num_tries_till_failure -= 1
    node_idx = choice(nodes)
    edge_idx = choice(edges)
    if edge_idx not in hypergraph.node[node_idx].edges:
      samples.add((node_idx, edge_idx))

  if len(samples) < num_samples:
    log.critical(
        "SampleMissingConnections failed to find %i samples",
        num_samples)
  return list(samples)


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
    links,
    distance_function=cosine,
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
    - links: the set of node_idx, edge_idx pairs to evaluate
  output:
    - list of node-edge pairs from links predicted to be accurate
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

  log.info("Calculating whether each point is in each edge's range")
  with Pool(num_cores,
            initializer=_init_get_missing_links,
            initargs=(node2embedding,
                      edge2centroid,
                      edge2range,
                      hypergraph,
                      distance_function)) as pool:
    for res in pool.imap(_get_missing_links, links, chunksize=250):
      if res:
        predicted_links.append(res)

  return predicted_links


def CalculateCommunityPredictionMetrics(
    predicted_connections,
    good_links,
    bad_links):
  """
    Treats good_links as positive examples, and computes a Metrics
    named tuple.
    input:
      - predicted_connections: iterable of (node_idx, edge_idx) pairs
      - good_links: iterable of (node_idx, edge_idx) pairs from original
      - bad_links: iterable of (node_idx, edge_idx) not from original
    output:
      - EvaluationMetrics proto containing all fields but accuracy
        and num_true_neg
    """
  predictions = set(predicted_connections)
  positives = set(good_links)
  negatives = set(bad_links)

  # + and - must be disjoin
  assert len(positives.intersection(negatives)) == 0

  # predictions must be a subset of the + and - samples
  assert len(predictions.intersection(
      positives.union(negatives))) == len(predictions)

  assert len(positives) + len(negatives) > 0

  true_positives = predictions.intersection(positives)

  metrics = EvaluationMetrics()

  if len(predictions):
    metrics.precision = len(true_positives) / len(predictions)

  if len(positives):
    metrics.recall = len(true_positives) / len(positives)

  if metrics.precision + metrics.recall:
    metrics.f1 = 2 * metrics.precision * metrics.recall / (
        metrics.precision + metrics.recall)

  metrics.num_true_pos = len(true_positives)
  metrics.num_false_pos = len(predictions) - len(true_positives)
  metrics.num_false_neg = len(positives) - len(true_positives)
  metrics.num_true_neg = len(negatives - predictions)
  metrics.accuracy = (metrics.num_true_pos + metrics.num_true_neg) / (
      len(positives) + len(negatives))
  return metrics
