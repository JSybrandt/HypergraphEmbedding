import unittest
from hypergraph_embedding import Hypergraph
from hypergraph_embedding.hypergraph_util import *
from hypergraph_embedding.embedding import *
import scipy as sp
import itertools


def TestHypergraph():
  h = Hypergraph()
  AddNodeToEdge(h, 0, 0)
  AddNodeToEdge(h, 1, 0)
  AddNodeToEdge(h, 1, 1)
  AddNodeToEdge(h, 2, 1)
  AddNodeToEdge(h, 2, 2)
  AddNodeToEdge(h, 3, 2)
  return h


class SvdEmbeddingTest(unittest.TestCase):

  def check_embedding(
      self,
      embedding,
      expected_idx,
      expected_len,
      expected_distinct=True):
    "Helper function to check that embedding is reasonable"
    "embedding - embedding object"
    "expected_idx - array of indices"
    "expected_len - int"
    "expected_distinct - if true, check that all vectors are different"

    self.assertEquals(len(embedding), len(expected_idx))
    for _, vec in embedding.items():
      self.assertEquals(len(vec), expected_len)

    if expected_distinct:
      for i, j in itertools.combinations(expected_idx, 2):
        self.assertNotEquals(list(embedding[i]), list(embedding[j]))

  def test_typical(self):
    "SvdEmbedding for test graph should create 3 distinct 2d vectors"
    dim = 2
    _input = TestHypergraph()
    actual = SvdEmbedding(dim)
    actual.embed(_input)
    self.check_embedding(actual, [0, 1, 2, 3], dim)

  def test_fail_dim_ge_nodes(self):
    "SvdEmbedding fails if the requested dimensionality is greater or"
    " equal to the number of nodes"
    _input = CreateRandomHyperGraph(num_nodes=5, num_edges=10, probability=1)
    actual = SvdEmbedding(5)
    with self.assertRaises(AssertionError):
      # Fails: dim = num nodes
      actual.embed(_input)
    actual = SvdEmbedding(6)
    with self.assertRaises(AssertionError):
      # Fails: dim > num nodes
      actual.embed(_input)
