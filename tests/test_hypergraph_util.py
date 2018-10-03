import unittest
from hypergraph_embedding import Hypergraph
from hypergraph_embedding.hypergraph_util import *
import scipy as sp
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from random import random
from random import randint


def EmptyHypergraph():
  return Hypergraph()


def TypicalHypergraph():
  h = Hypergraph()


def SparseArrayEquals(test, actual, expected):
  test.assertEqual(actual.shape, expected.shape)
  test.assertEqual((actual != expected).nnz, 0)


class HypergraphUtilTest(unittest.TestCase):

  def test_RemoveNodeFromEdge_typical(self):
    actual = Hypergraph()
    AddNodeToEdge(actual, 1, 2)
    AddNodeToEdge(actual, 1, 3)

    RemoveNodeFromEdge(actual, 1, 3)

    expected = Hypergraph()
    AddNodeToEdge(expected, 1, 2)

    self.assertEqual(actual, expected)

  def test_AddNodeToEdge_typical(self):
    "Adds both the node and the edge reference"
    actual = Hypergraph()
    AddNodeToEdge(actual, 1, 2)
    expected = Hypergraph()
    expected.node[1].edges.append(2)
    expected.edge[2].nodes.append(1)
    self.assertEqual(actual, expected)

  def test_AddNodeToEdge_dupl(self):
    "Duplicate calls to AddNodeToEdge should not effect the structure."
    actual = Hypergraph()
    AddNodeToEdge(actual, 1, 2)
    AddNodeToEdge(actual, 1, 2)
    expected = Hypergraph()
    expected.node[1].edges.append(2)
    expected.edge[2].nodes.append(1)
    self.assertEqual(actual, expected)

  def test_AddNodeToEdge_names(self):
    "Filling in name parameters should set hypergraph data names."
    actual = Hypergraph()
    AddNodeToEdge(actual, 0, 0, "A", "X")
    AddNodeToEdge(actual, 1, 0, node_name="B")
    AddNodeToEdge(actual, 1, 1, edge_name="Y")

    expected = Hypergraph()
    node_a = expected.node[0]
    node_b = expected.node[1]
    edge_x = expected.edge[0]
    edge_y = expected.edge[1]

    node_a.edges.append(0)
    node_a.name = "A"

    node_b.edges.append(0)
    node_b.edges.append(1)
    node_b.name = "B"

    edge_x.nodes.append(0)
    edge_x.nodes.append(1)
    edge_x.name = "X"

    edge_y.nodes.append(1)
    edge_y.name = "Y"

    self.assertEqual(actual, expected)

  def test_CreateRandomHyperGraph_k10(self):
    "Creates a 10x10 fully connected hypergraph"
    actual = CreateRandomHyperGraph(10, 10, 1)
    self.assertEqual(len(actual.node), 10)
    self.assertEqual(len(actual.edge), 10)
    for _, node in actual.node.items():
      self.assertEqual(len(node.edges), 10)
    for _, edge in actual.edge.items():
      self.assertEqual(len(edge.nodes), 10)

  def test_CreateRandomHyperGraph_empty(self):
    "Creates an empty hypergraph"
    actual = CreateRandomHyperGraph(10, 10, 0)
    expected = Hypergraph()
    self.assertEqual(actual, expected)

  def test_FromCsrMatrix_typical(self):
    "Nonzero in row i and col j means node i belong to edge j"
    _input = csr_matrix([
        [1, 0],  # node 0 in edge 0
        [1, 0],  # node 1 in edge 0
        [0, 1],  # node 2 in edge 1
        [1, 1]  # node 3 in edge 0 and 1
    ])
    actual = FromSparseMatrix(_input)
    expected = Hypergraph()
    expected.node[0].edges.append(0)
    expected.node[1].edges.append(0)
    expected.node[2].edges.append(1)
    expected.node[3].edges.append(0)
    expected.node[3].edges.append(1)
    expected.edge[0].nodes.append(0)
    expected.edge[0].nodes.append(1)
    expected.edge[0].nodes.append(3)
    expected.edge[1].nodes.append(2)
    expected.edge[1].nodes.append(3)
    self.assertEqual(actual, expected)

  def test_FromCsrMatrix_empty(self):
    "If the matrix is empty, return empty"
    actual = FromSparseMatrix(csr_matrix([]))
    expected = Hypergraph()
    self.assertEqual(actual, expected)

  def test_ToCsrMatrix_one(self):
    "Node i in edge j appears as a 1 in row i and col j"
    _input = Hypergraph()
    AddNodeToEdge(_input, 1, 2)
    actual = ToCsrMatrix(_input)
    expected = csr_matrix(
        [
            [0, 0, 0],  # node 0 not listed
            [0, 0, 1]  # node 1 in edge 2
        ],
        dtype=np.bool)
    SparseArrayEquals(self, actual, expected)

  def test_ToEdgeCsrMatrix_one(self):
    "Node i in edge j appears as a 1 in row j and col i"
    _input = Hypergraph()
    AddNodeToEdge(_input, 1, 2)
    actual = ToEdgeCsrMatrix(_input)
    expected = csr_matrix(
        [
            [0, 0],  # node 0 not listed
            [0, 0],
            [0, 1]  # node 1 in edge 2
        ],
        dtype=np.bool)
    SparseArrayEquals(self, actual, expected)

  def test_ToCsrMatrix_multiple(self):
    "Converting to csr handles multple nodes and multiple edges"
    _input = Hypergraph()
    AddNodeToEdge(_input, 1, 1)
    AddNodeToEdge(_input, 1, 2)
    AddNodeToEdge(_input, 2, 0)
    actual = ToCsrMatrix(_input)
    expected = csr_matrix(
        [
            [0, 0, 0],  # node 0 not listed
            [0, 1, 1],  # node 1 in edge 1 & 2
            [1, 0, 0]  # node 2 in edge 0
        ],
        dtype=np.bool)
    SparseArrayEquals(self, actual, expected)

  def test_ToCsrMatrix_empty(self):
    "If the hypergraph is empty, give me an empty matrix"
    _input = Hypergraph()
    actual = ToCsrMatrix(_input)
    expected = csr_matrix([])
    SparseArrayEquals(self, actual, expected)

  def test_ToFromCsrMatrix_fuzz(self):
    "any hypergraph should be preserved if converted to Csr and back"
    for i in range(100):
      num_nodes = randint(0, 10)
      num_edges = randint(0, 10)
      prob = random()
      hypergraph = CreateRandomHyperGraph(num_nodes, num_edges, prob)
      self.assertEqual(hypergraph, FromSparseMatrix(ToCsrMatrix(hypergraph)))

  def test_ToFromCsr_large_empty_graph(self):
    hypergraph = CreateRandomHyperGraph(100, 100, 0)
    self.assertEqual(hypergraph, FromSparseMatrix(ToCsrMatrix(hypergraph)))

  # CHANGE START HERE

  def test_FromCscMatrix_typical(self):
    "Nonzero in row i and col j means node i belong to edge j"
    _input = csr_matrix([
        [1, 0],  # node 0 in edge 0
        [1, 0],  # node 1 in edge 0
        [0, 1],  # node 2 in edge 1
        [1, 1]  # node 3 in edge 0 and 1
    ])
    actual = FromSparseMatrix(_input)
    expected = Hypergraph()
    expected.node[0].edges.append(0)
    expected.node[1].edges.append(0)
    expected.node[2].edges.append(1)
    expected.node[3].edges.append(0)
    expected.node[3].edges.append(1)
    expected.edge[0].nodes.append(0)
    expected.edge[0].nodes.append(1)
    expected.edge[0].nodes.append(3)
    expected.edge[1].nodes.append(2)
    expected.edge[1].nodes.append(3)
    self.assertEqual(actual, expected)

  def test_FromCscMatrix_empty(self):
    "If the matrix is empty, return empty"
    actual = FromSparseMatrix(csr_matrix([]))
    expected = Hypergraph()
    self.assertEqual(actual, expected)

  def test_ToCscMatrix_one(self):
    "Node i in edge j appears as a 1 in row i and col j"
    _input = Hypergraph()
    AddNodeToEdge(_input, 1, 2)
    actual = ToCscMatrix(_input)
    expected = csr_matrix(
        [
            [0, 0, 0],  # node 0 not listed
            [0, 0, 1]  # node 1 in edge 2
        ],
        dtype=np.bool)
    SparseArrayEquals(self, actual, expected)

  def test_ToCscMatrix_multiple(self):
    "Converting to csr handles multple nodes and multiple edges"
    _input = Hypergraph()
    AddNodeToEdge(_input, 1, 1)
    AddNodeToEdge(_input, 1, 2)
    AddNodeToEdge(_input, 2, 0)
    actual = ToCscMatrix(_input)
    expected = csr_matrix(
        [
            [0, 0, 0],  # node 0 not listed
            [0, 1, 1],  # node 1 in edge 1 & 2
            [1, 0, 0]  # node 2 in edge 0
        ],
        dtype=np.bool)
    SparseArrayEquals(self, actual, expected)

  def test_ToCscMatrix_empty(self):
    "If the hypergraph is empty, give me an empty matrix"
    _input = Hypergraph()
    actual = ToCscMatrix(_input)
    expected = csr_matrix([])
    SparseArrayEquals(self, actual, expected)

  def test_ToFromCscMatrix_fuzz(self):
    "any hypergraph should be preserved if converted to Csr and back"
    for i in range(100):
      num_nodes = randint(0, 10)
      num_edges = randint(0, 10)
      prob = random()
      hypergraph = CreateRandomHyperGraph(num_nodes, num_edges, prob)
      self.assertEqual(hypergraph, FromSparseMatrix(ToCscMatrix(hypergraph)))

  def test_ToFromCsc_large_empty_graph(self):
    hypergraph = CreateRandomHyperGraph(100, 100, 0)
    self.assertEqual(hypergraph, FromSparseMatrix(ToCscMatrix(hypergraph)))

  # CHANGE END HERE

  def test_ToBipartideNxGraph_typical(self):
    "ToBipartideNxGraph should handle a typical example. Edges become nodes"
    _input = Hypergraph()
    AddNodeToEdge(_input, 0, 0)
    AddNodeToEdge(_input, 1, 0)
    AddNodeToEdge(_input, 1, 1)
    AddNodeToEdge(_input, 2, 1)

    actual = ToBipartideNxGraph(_input)
    expected = nx.Graph()
    expected.add_edge(0, 3)  # community 0 from hypergraph becomes node 3
    expected.add_edge(1, 3)
    expected.add_edge(1, 4)
    expected.add_edge(2, 4)

    self.assertTrue(nx.is_isomorphic(actual, expected))

  def test_ToBipartideNxGraph_empty(self):
    "ToBipartideNxGraph should handle an empty example"
    actual = ToBipartideNxGraph(Hypergraph())
    expected = nx.Graph()
    self.assertTrue(nx.is_isomorphic(actual, expected))

  def test_ToCliqueNxGraph_empty(self):
    "ToCliqueNxGraph should handle an empty example"
    actual = ToCliqueNxGraph(Hypergraph())
    expected = nx.Graph()
    self.assertTrue(nx.is_isomorphic(actual, expected))

  def test_ToCliqueNxGraph_typical(self):
    "ToCliqueNxGraph should handle a small typical example"
    _input = Hypergraph()
    AddNodeToEdge(_input, 0, 0)
    AddNodeToEdge(_input, 1, 0)
    AddNodeToEdge(_input, 1, 1)
    AddNodeToEdge(_input, 2, 1)
    actual = ToCliqueNxGraph(_input)

    expected = nx.Graph()
    expected.add_edge(0, 1)
    expected.add_edge(1, 2)

    self.assertTrue(nx.is_isomorphic(actual, expected))

  def test_ToCliqueNxGraph_k4(self):
    "ToCliqueNxGraph should connect larger communities"
    _input = Hypergraph()
    AddNodeToEdge(_input, 0, 0)
    AddNodeToEdge(_input, 1, 0)
    AddNodeToEdge(_input, 2, 0)
    AddNodeToEdge(_input, 3, 0)
    actual = ToCliqueNxGraph(_input)

    expected = nx.Graph()
    expected.add_edge(0, 1)
    expected.add_edge(0, 2)
    expected.add_edge(0, 3)
    expected.add_edge(1, 2)
    expected.add_edge(1, 3)
    expected.add_edge(2, 3)

    self.assertTrue(nx.is_isomorphic(actual, expected))


class RelabelAndCompressionTest(unittest.TestCase):

  def test_typical(self):
    _input = Hypergraph()
    AddNodeToEdge(_input, 0, 1)
    AddNodeToEdge(_input, 1, 1)
    AddNodeToEdge(_input, 1, 2)

    node_map = {0: 100, 1: 200}
    edge_map = {1: 50, 2: 150}

    actual = Relabel(_input, node_map, edge_map)

    expected = Hypergraph()
    AddNodeToEdge(expected, 100, 50)
    AddNodeToEdge(expected, 200, 50)
    AddNodeToEdge(expected, 200, 150)

    self.assertEqual(actual, expected)

  def test_compress_range(self):
    original = Hypergraph()
    AddNodeToEdge(original, 100, 50)
    AddNodeToEdge(original, 200, 50)
    AddNodeToEdge(original, 200, 150)

    compressed, node_map, edge_map = CompressRange(original)
    restored = Relabel(compressed, node_map, edge_map)
    SparseArrayEquals(self, ToCsrMatrix(original), ToCsrMatrix(restored))
    self.assertEqual(len(compressed.node), max(compressed.node) + 1)
    self.assertEqual(len(compressed.node), len(original.node))
    self.assertEqual(len(compressed.edge), max(compressed.edge) + 1)
    self.assertEqual(len(compressed.edge), len(original.edge))

  def test_compress_range_fuzz(self):
    for _ in range(10):
      original = CreateRandomHyperGraph(100, 100, 0.01)
      compressed, node_map, edge_map = CompressRange(original)
      restored = Relabel(compressed, node_map, edge_map)
      SparseArrayEquals(self, ToCsrMatrix(original), ToCsrMatrix(restored))
      self.assertEqual(len(compressed.node), max(compressed.node) + 1)
      self.assertEqual(len(compressed.node), len(original.node))
      self.assertEqual(len(compressed.edge), max(compressed.edge) + 1)
      self.assertEqual(len(compressed.edge), len(original.edge))

class ToBlockDiagonalTest(unittest.TestCase):
  def test_typical(self):
    original = Hypergraph()
    AddNodeToEdge(original, 100, 50)
    AddNodeToEdge(original, 200, 50)
    AddNodeToEdge(original, 200, 150)

    compressed, node_map, edge_map = ToBlockDiagonal(original)
    self.assertEqual(len(compressed.node), max(compressed.node) + 1)
    self.assertEqual(len(compressed.node), len(original.node))
    self.assertEqual(len(compressed.edge), max(compressed.edge) + 1)
    self.assertEqual(len(compressed.edge), len(original.edge))

    restored = Relabel(compressed, node_map, edge_map)
    SparseArrayEquals(self, ToCsrMatrix(original), ToCsrMatrix(restored))

  def test_fuzz(self):
    for _ in range(10):
      original = CreateRandomHyperGraph(100, 100, 0.01)
      compressed, node_map, edge_map = ToBlockDiagonal(original)
      self.assertEqual(len(compressed.node), max(compressed.node) + 1)
      self.assertEqual(len(compressed.node), len(original.node))
      self.assertEqual(len(compressed.edge), max(compressed.edge) + 1)
      self.assertEqual(len(compressed.edge), len(original.edge))

      restored = Relabel(compressed, node_map, edge_map)
      SparseArrayEquals(self, ToCsrMatrix(original), ToCsrMatrix(restored))

class TestRemoveNode(unittest.TestCase):
  def test_typical(self):
    actual = Hypergraph()
    AddNodeToEdge(actual, 0, 0) # remove me
    AddNodeToEdge(actual, 0, 1) # remove me
    AddNodeToEdge(actual, 1, 0)
    AddNodeToEdge(actual, 1, 1)
    RemoveNode(actual, 0)

    expected = Hypergraph()
    AddNodeToEdge(expected, 1, 0)
    AddNodeToEdge(expected, 1, 1)
    self.assertEqual(actual, expected)

  def test_removes_edge(self):
    "Removing a node should automatically remove degree 0 edges"
    actual = Hypergraph()
    AddNodeToEdge(actual, 0, 0) # remove me
    AddNodeToEdge(actual, 0, 1) # remove me
    AddNodeToEdge(actual, 1, 0)
    RemoveNode(actual, 0)

    expected = Hypergraph()
    AddNodeToEdge(expected, 1, 0)
    self.assertEqual(actual, expected)

class TestRemoveEdge(unittest.TestCase):
  def test_typical(self):
    actual = Hypergraph()
    AddNodeToEdge(actual, 0, 0) # remove me
    AddNodeToEdge(actual, 0, 1)
    AddNodeToEdge(actual, 1, 0) # remove me
    AddNodeToEdge(actual, 1, 1)
    RemoveEdge(actual, 0)

    expected = Hypergraph()
    AddNodeToEdge(expected, 0, 1)
    AddNodeToEdge(expected, 1, 1)
    self.assertEqual(actual, expected)

  def test_removes_node(self):
    "Removing an edge should automatically remove degree 0 nodes"
    actual = Hypergraph()
    AddNodeToEdge(actual, 0, 0) # remove me
    AddNodeToEdge(actual, 0, 1)
    AddNodeToEdge(actual, 1, 0) # remove me
    RemoveEdge(actual, 0)

    expected = Hypergraph()
    AddNodeToEdge(expected, 0, 1)
    self.assertEqual(actual, expected)



if __name__ == "__main__":
  unittest.main()
