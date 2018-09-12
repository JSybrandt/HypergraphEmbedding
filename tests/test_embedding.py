import unittest
from hypergraph_embedding import Hypergraph
from hypergraph_embedding import HypergraphEmbedding
from hypergraph_embedding.hypergraph_util import *
from hypergraph_embedding.embedding import *
from hypergraph_embedding.hypergraph2vec import *
import scipy as sp
import itertools
from random import random, randint


def TestHypergraph():
  h = Hypergraph()
  AddNodeToEdge(h, 0, 0)
  AddNodeToEdge(h, 1, 0)
  AddNodeToEdge(h, 1, 1)
  AddNodeToEdge(h, 2, 1)
  AddNodeToEdge(h, 2, 2)
  AddNodeToEdge(h, 3, 2)
  return h


class EmbeddingTestCase(unittest.TestCase):

  def assertIndicesPresent(self, actual_embedding_dict, expected_indices):
    for i in expected_indices:
      self.assertTrue(i in actual_embedding_dict)

  def assertDim(self, actual_embedding_dict, expected_dim):
    for _, vec in actual_embedding_dict.items():
      self.assertEqual(len(vec.values), expected_dim)

  def assertNonZero(self, actual_embedding_dict):
    for _, vec in actual_embedding_dict.items():
      zero = True
      for f in vec.values:
        if f != 0:
          zero = False
          break
      self.assertFalse(zero)

  def checkEmbedding(self, embedding, hypergraph, dim):
    # check that dim property was set
    self.assertEqual(embedding.dim, dim)
    self.assertIndicesPresent(embedding.node, hypergraph.node)
    self.assertIndicesPresent(embedding.edge, hypergraph.edge)
    # check that all vectors actually have correct dim
    self.assertDim(embedding.node, dim)
    self.assertDim(embedding.edge, dim)
    # self.assertNonZero(embedding.node)
    # self.assertNonZero(embedding.edge)

  def help_test_fuzz(self, embedding_function, num_fuzz=100):
    "Random embedding should never break"
    for i in range(num_fuzz):
      hypergraph = CreateRandomHyperGraph(
          randint(1,
                  10),
          randint(1,
                  10),
          random())
      max_dim = min(len(hypergraph.node), len(hypergraph.edge))
      if max_dim <= 1:
        continue  # the random creation might not have actually made a hg
      dim = randint(1, max_dim - 1)
      actual = embedding_function(hypergraph, dim)
      self.checkEmbedding(actual, hypergraph, dim)


class EmbedSvdTest(EmbeddingTestCase):

  def test_typical(self):
    "SvdEmbedding for test graph should create 3 distinct 2d vectors"
    dim = 2
    _input = TestHypergraph()
    actual = EmbedSvd(_input, dim)
    self.checkEmbedding(actual, _input, dim)
    self.assertEqual(actual.method_name, "SVD")

  def test_fail_dim_ge_nodes(self):
    "SvdEmbedding fails if the requested dimensionality is greater or"
    " equal to the number of nodes"
    _input = CreateRandomHyperGraph(num_nodes=5, num_edges=10, probability=1)
    with self.assertRaises(AssertionError):
      # Fails: dim = num nodes
      EmbedSvd(_input, 5)
    with self.assertRaises(AssertionError):
      # Fails: dim > num nodes
      EmbedSvd(_input, 6)

  def test_fuzz(self):
    "EmbedSvd embedding should never break with valid input"
    self.help_test_fuzz(EmbedSvd)


class EmbedRandomTest(EmbeddingTestCase):

  def test_typical(self):
    "Random embedding should create 3 distinct 2d vectors for test graph"
    dim = 2
    _input = TestHypergraph()
    actual = EmbedRandom(_input, dim)
    self.checkEmbedding(actual, _input, dim)
    self.assertEqual(actual.method_name, "Random")

  def test_fuzz(self):
    "Random embedding should never break"
    self.help_test_fuzz(EmbedRandom)


class EmbedNmfTest(EmbeddingTestCase):

  def test_typical(self):
    dim = 2
    _input = TestHypergraph()
    actual = EmbedNMF(_input, dim)
    self.checkEmbedding(actual, _input, dim)
    self.assertEqual(actual.method_name, "NMF")

  def test_fuzz(self):
    "Random embedding should never break"
    self.help_test_fuzz(EmbedNMF)


class EmbedNode2VecBipartideTest(EmbeddingTestCase):

  def test_typical(self):
    dim = 2
    _input = TestHypergraph()
    actual = EmbedNode2VecBipartide(_input, dim)
    self.checkEmbedding(actual, _input, dim)
    self.assertEqual(actual.method_name, "Node2VecBipartide(5)")

  def test_fuzz(self):
    "Random embedding should never break"
    self.help_test_fuzz(EmbedNode2VecBipartide, num_fuzz=10)

  def test_disconnected_node(self):
    "Make sure we don't break if we have a totally disconnected node"
    dim = 2
    hg = Hypergraph()
    AddNodeToEdge(hg, 0, 0)
    AddNodeToEdge(hg, 1, 0)
    AddNodeToEdge(hg, 2, 1)

    actual = EmbedNode2VecBipartide(hg, dim)
    self.checkEmbedding(actual, hg, dim)
    self.assertEqual(actual.method_name, "Node2VecBipartide(5)")


class EmbedNode2VecCliqueTest(EmbeddingTestCase):

  def test_typical(self):
    dim = 2
    _input = TestHypergraph()
    actual = EmbedNode2VecClique(_input, dim)
    self.checkEmbedding(actual, _input, dim)
    self.assertEqual(actual.method_name, "Node2VecClique(5)")

  def test_fuzz(self):
    "Random embedding should never break"
    self.help_test_fuzz(EmbedNode2VecClique, num_fuzz=10)

  def test_disconnected_node(self):
    "Make sure we don't break if we have a totally disconnected node"
    dim = 2
    hg = Hypergraph()
    AddNodeToEdge(hg, 0, 0)
    AddNodeToEdge(hg, 1, 0)
    AddNodeToEdge(hg, 2, 1)

    actual = EmbedNode2VecClique(hg, dim)
    self.checkEmbedding(actual, hg, dim)
    self.assertEqual(actual.method_name, "Node2VecClique(5)")


class EmbedHypergraphTest(EmbeddingTestCase):

  def test_typical(self):
    dim = 2
    _input = TestHypergraph()
    actual = EmbedHypergraph(
        _input,
        dim,
        num_neighbors=2,
        pos_samples=2,
        neg_samples=1,
        batch_size=1,
        epochs=1)
    self.checkEmbedding(actual, _input, dim)
    self.assertEqual(actual.method_name, "Hypergraph")

  def test_fuzz(self):
    "Random embedding should never break"
    embed = lambda x, y: EmbedHypergraph(x,
                                         y,
                                         num_neighbors=2,
                                         pos_samples=2,
                                         neg_samples=1,
                                         batch_size=1,
                                         epochs=1)
    self.help_test_fuzz(embed, num_fuzz=10)


class EmbedHypergraphPlusPlusTest(EmbeddingTestCase):

  def test_typical(self):
    dim = 2
    _input = TestHypergraph()
    actual = EmbedHypergraphPlusPlus(
        _input,
        dim,
        num_neighbors=2,
        num_walks_per_node=2,
        max_walk_length=3,
        batch_size=1,
        epochs=1)
    self.checkEmbedding(actual, _input, dim)
    self.assertEqual(actual.method_name, "Hypergraph++")

  def test_fuzz(self):
    "Random embedding should never break"
    embed = lambda x, y: EmbedHypergraphPlusPlus(x,
                                         y,
                                         num_neighbors=2,
                                         num_walks_per_node=2,
                                         max_walk_length=3,
                                         batch_size=1,
                                         epochs=1)
    self.help_test_fuzz(embed, num_fuzz=10)

class EmbedAlgebraicDistanceTest(EmbeddingTestCase):

  def test_typical(self):
    dim = 2
    _input = TestHypergraph()
    actual = EmbedAlgebraicDistance(
        _input,
        dim)
    self.checkEmbedding(actual, _input, dim)
    self.assertEqual(actual.method_name, "AlgebraicDistance")
