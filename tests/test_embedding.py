import unittest
from hypergraph_embedding import Hypergraph
from hypergraph_embedding import HypergraphEmbedding
from hypergraph_embedding.hypergraph_util import *
from hypergraph_embedding.embedding import *
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


class EmbedSvdTest(EmbeddingTestCase):

  def test_typical(self):
    "SvdEmbedding for test graph should create 3 distinct 2d vectors"
    dim = 2
    _input = TestHypergraph()
    actual = EmbedSvd(_input, dim)
    self.checkEmbedding(actual, _input, dim)

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
    for i in range(100):
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
      actual = EmbedSvd(hypergraph, dim)
      self.checkEmbedding(actual, hypergraph, dim)


class EmbedRandomTest(EmbeddingTestCase):

  def test_typical(self):
    "Random embedding should create 3 distinct 2d vectors for test graph"
    dim = 2
    _input = TestHypergraph()
    actual = EmbedRandom(_input, dim)
    self.checkEmbedding(actual, _input, dim)

  def test_fuzz(self):
    "Random embedding should never break"
    for i in range(100):
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
      actual = EmbedRandom(hypergraph, dim)
      self.checkEmbedding(actual, hypergraph, dim)
