import unittest
from hypergraph_embedding import Hypergraph
from hypergraph_embedding.hypergraph_util import *

def EmptyHypergraph():
    return Hypergraph()

def TypicalHypergraph():
    h = Hypergraph()

class HypergraphUtilFunctions(unittest.TestCase):
    def test_AddNodeToEdge_typical(self):
        actual = Hypergraph()
        AddNodeToEdge(actual, 1, 2)
        expected = Hypergraph()
        expected.node[1].edges.append(2)
        expected.edge[2].nodes.append(1)
        self.assertEqual(str(actual), str(expected))

    def test_AddNodeToEdge_dupl(self):
        """
Duplicate calls to AddNodeToEdge should not effect the structure.
        """
        actual = Hypergraph()
        AddNodeToEdge(actual, 1, 2)
        AddNodeToEdge(actual, 1, 2)
        expected = Hypergraph()
        expected.node[1].edges.append(2)
        expected.edge[2].nodes.append(1)
        self.assertEqual(str(actual), str(expected))

    def test_CreateRandomHyperGraph_k10(self):
        """
        Creates a 10x10 fully connected hypergraph
        """
        actual = CreateRandomHyperGraph(10, 10, 1)
        self.assertEqual(len(actual.node), 10)
        self.assertEqual(len(actual.edge), 10)
        for _, node in actual.node.items():
            self.assertEqual(len(node.edges), 10)
        for _, edge in actual.edge.items():
            self.assertEqual(len(edge.nodes), 10)

    def test_CreateRandomHyperGraph_empty(self):
        """
        Creates an empty hypergraph
        """
        actual = CreateRandomHyperGraph(10, 10, 0)
        self.assertEqual(len(actual.node), 0)
        self.assertEqual(len(actual.edge), 0)

    def test_ToSparseMatrix(self):
        self.assertEqual(ToSparseMatrix(EmptyHypergraph()), 1)

if __name__ == "__main__":
    unittest.main()
