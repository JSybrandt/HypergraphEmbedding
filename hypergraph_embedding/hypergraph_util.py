# This file contains functions for generating and manipulating the Hypergraph
# proto message.

from .hypergraph_pb2 import Hypergraph
import numpy as np
import scipy as sp
import networkx as nx
from random import random
import itertools
import logging


def AddNodeToEdge(hypergraph, node_id, edge_id, node_name=None, edge_name=None):
  """
    Modifies hypergraph by setting a connection from given node to given edge.
    If node/edge name are supplied.
    """
  assert node_id >= 0
  assert edge_id >= 0

  node = hypergraph.node[node_id]
  edge = hypergraph.edge[edge_id]

  if edge_id not in node.edges:
    node.edges.append(edge_id)
  if node_id not in edge.nodes:
    edge.nodes.append(node_id)
  if node_name is not None:
    if node.HasField("name") and node.name != node_name:
      logging.getLogger().warning(
          "Overwriting Node #{} name from {} to {}".format(
              node_id, node.name, node_name))
    node.name = node_name
  if edge_name is not None:
    if edge.HasField("name") and edge.name != edge_name:
      logging.getLogger().warning(
          "Overwriting Edge #{} name from {} to {}".format(
              edge_id, edge.name, edge_name))
    edge.name = edge_name
  return hypergraph


def CreateRandomHyperGraph(num_nodes, num_edges, probability):
  """
    Creates a graph of `num_nodes` and `num_edges` where `probability` is the
    chance that node i belongs to edge j.
    """
  assert probability <= 1
  assert probability >= 0
  assert num_edges >= 0
  assert num_nodes >= 0
  result = Hypergraph()
  for i in range(num_nodes):
    for j in range(num_edges):
      if random() < probability:
        AddNodeToEdge(result, i, j)
  return result


def FromSparseMatrix(sparse_matrix):
  """
    Creates a hypergraph object from the provided sparse matrix. Each row
    represents a node, each column represents an edge. A 1 in row i and
    column j represents that node i belongs to edge j.
    """
  res = Hypergraph()
  rows, cols = sparse_matrix.nonzero()
  for r, c in zip(rows, cols):
    AddNodeToEdge(res, r, c)
  return res


def IsEmpty(hypergraph):
  "Returns true if there are no nodes or edges"
  return len(hypergraph.node) == 0 or len(hypergraph.edge) == 0


def ToCsrMatrix(hypergraph):
  """
    ToSparseMatrix accepts a hypergraph proto message and converts it to a
    Compressed Sparse Row matrix via scipy. Each row represents a node, each
    column represents an edge. A 1 in row i and column j represents that node i
    belongs to edge j.
    """
  if IsEmpty(hypergraph):
    # if the hypergraph is empty, return empty matrix
    return sp.sparse.csr_matrix([])
  vals = []
  rows = []
  cols = []
  for node_idx, node in hypergraph.node.items():
    for edge_idx in node.edges:
      vals.append(1)
      rows.append(node_idx)
      cols.append(edge_idx)
  return sp.sparse.csr_matrix((vals, (rows, cols)), dtype=np.bool)


def ToEdgeCsrMatrix(hypergraph):
  """
    ToSparseMatrix accepts a hypergraph proto message and converts it to a
    Compressed Sparse Row matrix via scipy. Each row represents a node, each
    column represents an edge. A 1 in row i and column j represents that node i
    belongs to edge j.
    """
  if IsEmpty(hypergraph):
    # if the hypergraph is empty, return empty matrix
    return sp.sparse.csr_matrix([])
  vals = []
  rows = []
  cols = []
  for edge_idx, edge in hypergraph.edge.items():
    for node_idx in edge.nodes:
      vals.append(1)
      rows.append(edge_idx)
      cols.append(node_idx)
  return sp.sparse.csr_matrix((vals, (rows, cols)), dtype=np.bool)


def ToCscMatrix(hypergraph):
  """
    ToSparseMatrix accepts a hypergraph proto message and converts it to a
    Compressed Sparse Column matrix via scipy. Each row represents a node, each
    column represents an edge. A 1 in row i and column j represents that node i
    belongs to edge j.
    """
  if IsEmpty(hypergraph):
    # if the hypergraph is empty, return empty matrix
    return sp.sparse.csc_matrix([])
  vals = []
  rows = []
  cols = []
  for node_idx, node in hypergraph.node.items():
    for edge_idx in node.edges:
      vals.append(1)
      rows.append(node_idx)
      cols.append(edge_idx)
  return sp.sparse.csc_matrix((vals, (rows, cols)), dtype=np.bool)


def ToBipartideNxGraph(hypergraph):
  """
    Converts the hypergraph into a networkx graph via the bipartide method.
    Each edge from the original hypergraph becomes a node (indexed starting at
    # nodes). Edges in the new graph represent community membership.
    """
  if IsEmpty(hypergraph):
    return nx.Graph()

  max_node_id = max([i for i, _ in hypergraph.node.items()])

  def hyperedge_graph_id(edge_num):
    "edges indexed 0 to #edges-1, becomes new ids after # nodes"
    return max_node_id + 1 + edge_num

  result = nx.Graph()
  for node_idx, node in hypergraph.node.items():
    for edge_idx in node.edges:
      result.add_edge(node_idx, hyperedge_graph_id(edge_idx))
  return result


def ToCliqueNxGraph(hypergraph):
  """
    Converts the hypergraph into a networkx graph via the clique method.
    Communities from the original graph are replaced with fully connected
    cliques in the result graph.
    """
  if IsEmpty(hypergraph):
    return nx.Graph()

  result = nx.Graph()
  for _, edge in hypergraph.edge.items():
    # iterate all pairs within hyperedge
    for i, j in itertools.combinations(edge.nodes, 2):
      result.add_edge(i, j)
  return result

def Relabel(original_hg, node_map, edge_map):
  """
  Given a hypergraph and optional maps of nodes/edges. Create a new hypergraph
  where nodes/edges have been relabeled.
  """

  relabed_hg = Hypergraph()
  for node_idx, node in original_hg.node.items():
    for edge_idx in node.edges:
      assert node_idx in node_map
      assert edge_idx in edge_map
      AddNodeToEdge(relabed_hg, node_map[node_idx], edge_map[edge_idx])

  return relabed_hg

def CompressRange(original_hg):
  """
  Given a hypergraph, produce a new object where all nodes / edges have been
  moved into the 0-n range. Additionally produces the node/edge map to restore
  the original.
  """

  compressed_hg = Hypergraph()

  node_map = {}
  edge_map = {}

  for node_idx, node in original_hg.node.items():
    if node_idx not in node_map:
      node_map[node_idx] = len(node_map)
    for edge_idx in node.edges:
      if edge_idx not in edge_map:
        edge_map[edge_idx] = len(edge_map)
      AddNodeToEdge(compressed_hg, node_map[node_idx], edge_map[edge_idx])

  inv_node_map = {y: x for x, y in node_map.items()}
  inv_edge_map = {y: x for x, y in edge_map.items()}

  return compressed_hg, inv_node_map, inv_edge_map
