# This file provides evaluation utilities for hypergraph experiments

from random import random
from . import Hypergraph
from .hypergraph_util import *
from scipy.spatial.distance import cosine
import numpy as np


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


def CommunityPrediction(hypergraph, embedding, distance_function=cosine):
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
  output:
    - list of missing node-edge pairs
  """

  assert embedding.dim > 0

  edge2centroid = {}
  edge2max_distance = {}

  for edge_idx, edge in hypergraph.edge.items():
    if len(edge.nodes) >= 2:
      points = []
      for node_idx in edge.nodes:
        points.append(np.asarray(embedding.node[node_idx].values))
      centroid = np.mean(points, axis=0)
      max_dist = max([distance_function(centroid, vec) for vec in points])
      edge2centroid[edge_idx] = centroid
      edge2max_distance[edge_idx] = max_dist

  missing_links = []
  for node_idx, embedding in embedding.node.items():
    vec = np.asarray(embedding.values)
    for edge_idx in hypergraph.edge:
      if distance_function(
          vec,
          edge2centroid[edge_idx]) <= edge2max_distance[edge_idx]:
        if edge_idx not in hypergraph.node[node_idx].edges:
          missing_links.append((node_idx, edge_idx))
  return missing_links
