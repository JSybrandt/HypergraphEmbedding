import unittest
from hypergraph_embedding import Hypergraph
from hypergraph_embedding import HypergraphEmbedding
from hypergraph_embedding.hypergraph_util import *
from hypergraph_embedding.weighted_hypergraph2vec import *
import hypergraph_embedding.weighted_hypergraph2vec as whg2v
from hypergraph_embedding.embedding import *
from random import random, randint, choice
from scipy.sparse import lil_matrix


class ComputeSpansTest(unittest.TestCase):

  def test_one_pair(self):
    "Given an embedding, find the distance to each node/edge first "
    "order neighbor"
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 3, 7)
    embedding = HypergraphEmbedding()
    embedding.node[3].values.extend([0])
    embedding.edge[7].values.extend([1])

    node2span, edge2span = ComputeSpans(
        hypergraph, embedding, disable_pbar=True)
    self.assertEqual(node2span, {3: 1})
    self.assertEqual(edge2span, {7: 1})

  def test_tripple(self):
    "Given an embedding, find the distance to each node/edge first "
    "order neighbor"
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 3, 7)
    AddNodeToEdge(hypergraph, 5, 7)
    embedding = HypergraphEmbedding()
    embedding.node[3].values.extend([0])
    embedding.node[5].values.extend([3])
    embedding.edge[7].values.extend([1])

    node2span, edge2span = ComputeSpans(
        hypergraph, embedding, disable_pbar=True)
    self.assertEqual(node2span, {3: 1, 5: 2})
    # This span goes from 0 to 3
    self.assertEqual(edge2span, {7: 3})

  def test_two_d(self):
    "Given 2d embeddings, we should be using SUPREMUM norm"

    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 10, 20)
    embedding = HypergraphEmbedding()
    embedding.node[10].values.extend([0, 0])
    embedding.edge[20].values.extend([3, 4])

    # Note the difference between 10 and 20 makes a 3-4-5 triangle
    node2span, edge2span = ComputeSpans(
        hypergraph, embedding, disable_pbar=True)
    self.assertEqual(node2span, {10: 4})
    self.assertEqual(edge2span, {20: 4})

  def test_no_link_zero(self):
    "If we have a node without connections, treat its span as 0"
    hypergraph = Hypergraph()
    hypergraph.node[1].name = "I shouldn't mess stuff up"
    hypergraph.edge[2].name = "I shouldn't mess stuff up"

    AddNodeToEdge(hypergraph, 0, 0)

    embedding = HypergraphEmbedding()
    embedding.node[1].values.extend([1, 2])
    embedding.edge[2].values.extend([3, 4])
    embedding.node[0].values.extend([0, 0])
    embedding.edge[0].values.extend([3, 4])

    node2span, edge2span = ComputeSpans(
        hypergraph, embedding, disable_pbar=True)
    self.assertEqual(node2span, {0: 4, 1: 0})
    self.assertEqual(edge2span, {0: 4, 2: 0})

  def test_fuzz(self):
    "Needs to make its own embedding if I don't supply one"
    for i in range(10):
      dim = 2
      hypergraph = CreateRandomHyperGraph(10, 10, 0.1)
      embedding = EmbedRandom(hypergraph, 2)
      # make sure it doesn't break
      node2rad, edge2rad = ComputeSpans(
          hypergraph, embedding, disable_pbar=True)
      for node_idx in hypergraph.node:
        self.assertTrue(node_idx in node2rad)
        self.assertTrue(node2rad[node_idx] >= 0)
      for edge_idx in hypergraph.edge:
        self.assertTrue(edge_idx in edge2rad)
        self.assertTrue(edge2rad[edge_idx] >= 0)


class ZeroOneScaleKeysTest(unittest.TestCase):

  def test_typical(self):
    _input = {0: 5, 1: 10}
    actual = ZeroOneScaleKeys(_input, run_in_parallel=False, disable_pbar=True)
    expected = {0: 0, 1: 1}
    self.assertEqual(actual, expected)

  def test_no_range(self):
    "in the case of only one element, set all values to 1"
    _input = {0: 5}
    actual = ZeroOneScaleKeys(_input, run_in_parallel=False, disable_pbar=True)
    expected = {0: 1}
    self.assertEqual(actual, expected)

  def test_empty_range(self):
    "Do nothing and don't crash if no input"
    _input = {}
    actual = ZeroOneScaleKeys(_input, run_in_parallel=False, disable_pbar=True)
    expected = {}
    self.assertEqual(actual, expected)

  def test_fuzz(self):
    for i in range(10):
      _input = {}
      for idx in range(randint(0, 20)):
        _input[idx] = random() * 10
      actual = ZeroOneScaleKeys(_input, disable_pbar=True)
      self.assertEqual(set(_input.keys()), set(actual.keys()))
      if len(_input) > 0:
        self.assertEqual(max(actual.values()), 1)
      if len(_input) > 1:
        self.assertEqual(min(actual.values()), 0)


class SameTypeProbabilityTest(unittest.TestCase):

  def test_typical(self):
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 1, 4)  # only 1 in 4
    AddNodeToEdge(hypergraph, 1, 5)  # 1 & 2 in 5
    AddNodeToEdge(hypergraph, 2, 5)
    AddNodeToEdge(hypergraph, 1, 6)  # 1 & 2 in 6
    AddNodeToEdge(hypergraph, 2, 6)
    AddNodeToEdge(hypergraph, 2, 7)  # only 2 in 7
    node2edges = ToCsrMatrix(hypergraph)
    alpha = 0.25
    edge2span = {
        4: 0.1,
        5: 0.2,
        6: 0.3,
        7: 0.4,
    }
    """
     (α + (1−α)(1−0.2))
    +(α + (1−α)(1−0.3))
     ------------------
     (α + (1−α)(1−0.1))
    +(α + (1−α)(1−0.2))
    +(α + (1−α)(1−0.3))
    +(α + (1−α)(1−0.4))
    = 0.5
    """
    actual = whg2v._same_type_probability((1,
                                           2),
                                          node2edges,
                                          alpha,
                                          edge2span)
    self.assertAlmostEqual(actual, 0.5)

  def test_zero(self):
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 10)
    AddNodeToEdge(hypergraph, 1, 20)
    node2edges = ToCsrMatrix(hypergraph)
    alpha = 0.25
    edge2span = {10: 0.4, 20: 0.8}
    actual = whg2v._same_type_probability((0,
                                           1),
                                          node2edges,
                                          alpha,
                                          edge2span)
    # These two nodes don't share anything in common
    self.assertAlmostEqual(actual, 0)

  def test_self_prob(self):
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 1)
    node2edges = ToCsrMatrix(hypergraph)
    alpha = 0.25
    edge2span = {1: 0.4}
    actual = whg2v._same_type_probability((0,
                                           0),
                                          node2edges,
                                          alpha,
                                          edge2span)
    # A thing should be equal to itself
    self.assertAlmostEqual(actual, 1)


class NodeEdgeProbabilityTest(unittest.TestCase):

  def test_typical(self):
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 1, 4)
    AddNodeToEdge(hypergraph, 2, 4)
    AddNodeToEdge(hypergraph, 2, 5)
    AddNodeToEdge(hypergraph, 3, 5)

    node2edge = ToCsrMatrix(hypergraph)
    edge2node = ToEdgeCsrMatrix(hypergraph)
    node2node = node2edge * node2edge.T
    edge2edge = edge2node * edge2node.T
    alpha = 0
    node2span = {1: 1, 2: 0, 3: 0.5}
    edge2span = {4: 0.5, 5: 0.5}
    actual = whg2v._node_edge_probability((1,
                                           5),
                                          node2edge,
                                          edge2node,
                                          node2node,
                                          edge2edge,
                                          alpha,
                                          node2span,
                                          edge2span)
    # it is almost too hard to explain how I got these numbers to come out
    self.assertAlmostEqual(actual, 0.7)

  def test_fuzz(self):
    for _ in range(10):
      hypergraph = Hypergraph()
      while len(hypergraph.node) == 0 or len(hypergraph.edge) == 0:
        hypergraph = CreateRandomHyperGraph(10, 10, 0.1)
      node2edge = ToCsrMatrix(hypergraph)
      edge2node = ToEdgeCsrMatrix(hypergraph)
      node2node = node2edge * node2edge.T
      edge2edge = edge2node * edge2node.T
      alpha = random()
      node2span = {i: random() for i in hypergraph.node}
      edge2span = {i: random() for i in hypergraph.edge}
      for _ in range(10):
        actual = whg2v._node_edge_probability(
            (choice(list(hypergraph.node)),
             choice(list(hypergraph.edge))),
            node2edge,
            edge2node,
            node2node,
            edge2edge,
            alpha,
            node2span,
            edge2span)
        self.assertTrue(actual >= 0)
        self.assertTrue(actual <= 1)


# class GetWeightedModelTest(unittest.TestCase):

# def test_no_break(self):
# hypergraph = CreateRandomHyperGraph(10, 10, 0.5)
# dimension = 100
# num_neighbors = 5
# model = GetWeightedModel(hypergraph, dimension, num_neighbors)
# model.summary()
# from keras.utils.vis_utils import plot_model
# plot_model(model, "model.png")
