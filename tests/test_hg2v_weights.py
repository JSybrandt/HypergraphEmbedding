#/usr/bin/env python3

import unittest
from hypergraph_embedding.hg2v_weighting import UniformWeight
from hypergraph_embedding.hg2v_weighting import DictToSparseRow
from hypergraph_embedding import Hypergraph
from hypergraph_embedding.hypergraph_util import AddNodeToEdge
from scipy.sparse import csr_matrix
import numpy as np


class SparseMatrixTestCase(unittest.TestCase):

  def assertSparseAlmostEqual(self, actual, expected, tol=1E-5):
    self.assertEqual(actual.shape, expected.shape)
    # >= is quicker than < apparently
    # if any element is >= tol, then np.max returns true
    self.assertFalse(np.max(np.abs(actual - expected) >= tol))


class UniformWeightTest(SparseMatrixTestCase):

  def test_typical(self):
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 1)
    AddNodeToEdge(hypergraph, 2, 2)
    AddNodeToEdge(hypergraph, 3, 2)

    actual_node2weight, actual_edge2weight = UniformWeight(hypergraph)

    expected_node2weight = csr_matrix([[0, 1, 0],# node 0 edge 1
                                       [0, 0, 0],# node 1 not present
                                       [0, 0, 1],# node 2 edge 2
                                       [0, 0, 1] # node 3 edge 2
                                      ], dtype=np.float32)
    expected_edge2weight = csr_matrix([[0, 0, 0, 0], # no edge 0
                                       [1, 0, 0, 0], # edge 1 node 0
                                       [0, 0, 1, 1]  # edge 2 node 2 and 3
                                      ], dtype=np.float32)
    self.assertSparseAlmostEqual(actual_node2weight, expected_node2weight)
    self.assertSparseAlmostEqual(actual_edge2weight, expected_edge2weight)


class DictToSparseRowTest(SparseMatrixTestCase):

  def test_typical(self):
    _input = {0: 1, 2: 4, 5: 100}
    actual = DictToSparseRow(_input)
    expected = csr_matrix([1, 0, 4, 0, 0, 100], dtype=np.float32)
    self.assertSparseAlmostEqual(actual, expected)
