import unittest
from hypergraph_embedding import Hypergraph
from hypergraph_embedding import HypergraphEmbedding
from hypergraph_embedding.data_util import *
from hypergraph_embedding.evaluation_util import *
from random import random, randint


class TestRemoveRandomConnections(unittest.TestCase):

  def checkSubset(self, removed, original, removed_list):
    "Ensures that all node-edge connections found in removed are present "
    "in original"
    node2edge = set()
    for node_idx, node in original.node.items():
      for edge_idx in node.edges:
        node2edge.add((node_idx, edge_idx))

    for node_idx, node in removed.node.items():
      for edge_idx in node.edges:
        # all node-edge connections should be found
        self.assertTrue((node_idx, edge_idx) in node2edge)
        node2edge.remove((node_idx, edge_idx))

    # all remaining connections should be represented in the removed list
    self.assertEqual(node2edge, set(removed_list))

  def checkValid(self, hypergraph):
    "Ensures that if node-edge is present, that edge-node is also"
    node2edge = set()
    for node_idx, node in hypergraph.node.items():
      for edge_idx in node.edges:
        node2edge.add((node_idx, edge_idx))
    for edge_idx, edge in hypergraph.edge.items():
      for node_idx in edge.nodes:
        # all node-edge connections must be found
        self.assertTrue((node_idx, edge_idx) in node2edge)
        node2edge.remove((node_idx, edge_idx))
    # no connections may be missing
    self.assertEqual(len(node2edge), 0)

  def test_remove_all(self):
    _input = Hypergraph()
    AddNodeToEdge(_input, 0, 0)
    AddNodeToEdge(_input, 0, 1)
    AddNodeToEdge(_input, 1, 1)
    actual_hg, removed_list = RemoveRandomConnections(_input, 1)
    self.assertEqual(actual_hg, Hypergraph())
    # better not remove the original
    self.assertNotEqual(actual_hg, _input)
    self.assertEqual(set(removed_list),
                     set([
                         (0,
                          0),
                         (0,
                          1),
                         (1,
                          1),
                     ]))

  def test_remove_none(self):
    _input = Hypergraph()
    AddNodeToEdge(_input, 0, 0)
    AddNodeToEdge(_input, 0, 1)
    AddNodeToEdge(_input, 1, 1)
    actual_hg, removed_list = RemoveRandomConnections(_input, 0)
    self.assertEqual(removed_list, [])
    self.checkSubset(actual_hg, _input, [])

  def test_keeps_names(self):
    _input = Hypergraph()
    AddNodeToEdge(_input, 0, 0, "A", "X")
    AddNodeToEdge(_input, 0, 1, "A", "Y")
    AddNodeToEdge(_input, 1, 1, "B", "Y")
    actual_hg, removed_list = RemoveRandomConnections(_input, 0)
    self.assertEqual(actual_hg, _input)
    self.checkSubset(actual_hg, _input, [])

  def test_fuzz(self):
    for i in range(100):
      num_nodes = randint(0, 10)
      num_edges = randint(0, 10)
      edge_prob = random()
      original = CreateRandomHyperGraph(num_nodes, num_edges, edge_prob)
      remove_prob = random()
      removed_hg, removed_list = RemoveRandomConnections(original, remove_prob)
      self.checkSubset(removed_hg, original, removed_list)
      self.checkValid(removed_hg)


class TestCommunityPrediction(unittest.TestCase):

  def test_typical(self):
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 0)
    AddNodeToEdge(hypergraph, 1, 0)

    embedding = HypergraphEmbedding()
    embedding.dim = 2
    embedding.node[0].values.extend([0, 1])
    embedding.node[1].values.extend([0, 0])
    embedding.node[2].values.extend([0, 0.5])

    actual = CommunityPrediction(hypergraph, embedding)
    self.assertEqual(set(actual), set([(2, 0)]))
