#/usr/bin/env python3

import unittest
from hypergraph_embedding.hg2v_sample import SimilarityRecord
from hypergraph_embedding.hg2v_sample import SamplesToModelInput
from hypergraph_embedding.hg2v_sample import SparseWeightedJaccard
from hypergraph_embedding.hg2v_sample import SameTypeJaccardSample
from hypergraph_embedding.hg2v_sample import CentroidFromRows
from scipy.sparse import csr_matrix
import numpy as np


class SamplesToModelInputTest(unittest.TestCase):

  def test_node_sim_unweighted(self):
    _input = SimilarityRecord(
        left_node_idx=0, right_node_idx=1, node_node_prob=0.5)
    actual = SamplesToModelInput([_input], num_neighbors=2, weighted=False)
    # note, the node indices should be incremented
    expected = (
        [
            [1],  # left_node_idx
            [0],  # left_edge_idx
            [2],  # right_node_idx
            [0],  # right_edge_idx
            [0],
            [0],  # neighbor_node_indices
            [0],
            [0]
        ],  # neighbor_edge_indices
        [
            [0.5],  # node_node_prob
            [0],  # edge_edge_prob
            [0]
        ])  # node_edge_prob
    self.assertEqual(actual, expected)

  def test_edge_sim_unweighted(self):
    _input = SimilarityRecord(
        left_edge_idx=0, right_edge_idx=1, edge_edge_prob=0.5)
    actual = SamplesToModelInput([_input], num_neighbors=2, weighted=False)
    # note, the node indices should be incremented
    expected = (
        [
            [0],  # left_node_idx
            [1],  # left_edge_idx
            [0],  # right_node_idx
            [2],  # right_edge_idx
            [0],
            [0],  # neighbor_node_indices
            [0],
            [0]
        ],  # neighbor_edge_indices
        [
            [0],  # node_node_prob
            [0.5],  # edge_edge_prob
            [0]
        ])  # node_edge_prob
    self.assertEqual(actual, expected)

  def test_edge_node_sim_unweighted(self):
    _input = SimilarityRecord(
        left_node_idx=0,
        right_edge_idx=1,
        neighbor_node_indices=[2],
        neighbor_edge_indices=[3, 4],
        node_edge_prob=0.5)
    actual = SamplesToModelInput([_input], num_neighbors=2, weighted=False)
    # note, the node indices should be incremented
    expected = (
        [
            [1],  # left_node_idx
            [0],  # left_edge_idx
            [0],  # right_node_idx
            [2],  # right_edge_idx
            [3],
            [0],  # neighbor_node_indices
            [4],
            [5]
        ],  # neighbor_edge_indices
        [
            [0],  # node_node_prob
            [0],  # edge_edge_prob
            [0.5]
        ])  # node_edge_prob
    self.assertEqual(actual, expected)

  def test_node_sim_weighted(self):
    _input = SimilarityRecord(
        left_node_idx=0,
        right_node_idx=1,
        left_weight=0.3,
        right_weight=0.6,
        node_node_prob=0.5)
    actual = SamplesToModelInput([_input], num_neighbors=2, weighted=True)
    # note, the node indices should be incremented
    expected = (
        [
            [1],  # left_node_idx
            [0],  # left_edge_idx
            [2],  # right_node_idx
            [0],  # right_edge_idx
            [0.3],  # left_weight
            [0.6],  # right_weight
            [0],
            [0],  # neighbor_node_indices
            [0],
            [0],  # neighbor_node_weights
            [0],
            [0],  # neighbor_edge_indices
            [0],
            [0]
        ],  # neighbor_edge_weights
        [
            [0.5],  # node_node_prob
            [0],  # edge_edge_prob
            [0]
        ])  # node_edge_prob
    self.assertEqual(actual, expected)

  def test_edge_sim_weighted(self):
    _input = SimilarityRecord(
        left_edge_idx=0,
        right_edge_idx=1,
        left_weight=0.3,
        right_weight=0.6,
        neighbor_node_indices=[2],
        neighbor_node_weights=[0.25],
        neighbor_edge_indices=[3, 4],
        neighbor_edge_weights=[0.5, 0.75],
        node_edge_prob=0.5)
    actual = SamplesToModelInput([_input], num_neighbors=2)
    # note, the node indices should be incremented
    expected = (
        [
            [0],  # left_node_idx
            [1],  # left_edge_idx
            [0],  # right_node_idx
            [2],  # right_edge_idx
            [0.3],  # left_weight
            [0.6],  # right_weight
            [3],
            [0],  # neighbor_node_indices
            [0.25],
            [0],  # neighbor_node_weights
            [4],
            [5],  # neighbor_edge_indices
            [0.5],
            [.75]
        ],  # neighbor_edge_weights
        [
            [0],  # node_node_prob
            [0],  # edge_edge_prob
            [0.5]
        ])  # node_edge_prob
    self.assertEqual(actual, expected)

  def test_node_edge_sim_weighted(self):
    _input = SimilarityRecord(
        left_edge_idx=0,
        right_edge_idx=1,
        left_weight=0.3,
        right_weight=0.6,
        edge_edge_prob=0.5)
    actual = SamplesToModelInput([_input], num_neighbors=2)
    # note, the node indices should be incremented
    expected = (
        [
            [0],  # left_node_idx
            [1],  # left_edge_idx
            [0],  # right_node_idx
            [2],  # right_edge_idx
            [0.3],  # left_weight
            [0.6],  # right_weight
            [0],
            [0],  # neighbor_node_indices
            [0],
            [0],  # neighbor_node_weights
            [0],
            [0],  # neighbor_edge_indices
            [0],
            [0]
        ],  # neighbor_edge_weights
        [
            [0],  # node_node_prob
            [0.5],  # edge_edge_prob
            [0]
        ])  # node_edge_prob
    self.assertEqual(actual, expected)


class SparseWeightedJaccardTest(unittest.TestCase):

  def test_boolean(self):
    row_i = csr_matrix([0, 1, 1, 0, 1], dtype=np.bool)
    row_j = csr_matrix([1, 0, 1, 0, 1], dtype=np.bool)
    actual = SparseWeightedJaccard(row_i, row_j)
    self.assertEqual(actual, 0.5)

  def test_typical(self):
    row_i = csr_matrix([0, 3, 4, 0, 1], dtype=np.int32)
    row_j = csr_matrix([2, 0, 1, 0, 1], dtype=np.int32)
    # min(2,0) + min(3,0) + min(4,1) + min(0,0) + min(1,1) = 2
    # ---------------------------------------------------    --
    # max(2,0) + max(3,0) + max(4,1) + max(0,0) + max(1,1) = 10
    actual = SparseWeightedJaccard(row_i, row_j)
    self.assertAlmostEqual(actual, 0.2)


class SampleTest(unittest.TestCase):

  def assertSimRecEq(self, a, b):
    self.assertEqual(a.left_node_idx, b.left_node_idx)
    self.assertEqual(a.left_edge_idx, b.left_edge_idx)
    self.assertEqual(a.right_node_idx, b.right_node_idx)
    self.assertEqual(a.right_edge_idx, b.right_edge_idx)
    self.assertAlmostEqual(a.left_weight, b.left_weight)
    self.assertAlmostEqual(a.right_weight, b.right_weight)
    self.assertEqual(a.neighbor_node_indices, b.neighbor_node_indices)
    self.assertEqual(a.neighbor_edge_indices, b.neighbor_edge_indices)
    self.assertAlmostEqual(a.neighbor_node_weights, b.neighbor_node_weights)
    self.assertAlmostEqual(a.neighbor_edge_weights, b.neighbor_edge_weights)
    self.assertAlmostEqual(a.node_node_prob, b.node_node_prob)
    self.assertAlmostEqual(a.edge_edge_prob, b.edge_edge_prob)
    self.assertAlmostEqual(a.node_edge_prob, b.node_edge_prob)

  def assertSimRecsMatch(self, recs_a, recs_b):
    "Attempts to match similarity records on keys"

    def key(rec):
      return (rec.left_node_idx, rec.left_edge_idx, rec.right_node_idx,
              rec.right_edge_idx)

    keyed_a = {key(a): a for a in recs_a}
    keyed_b = {key(b): b for b in recs_b}
    self.assertEqual(keyed_a.keys(), keyed_b.keys())
    for key in keyed_a:
      self.assertSimRecEq(keyed_a[key], keyed_b[key])


class SameTypeSampleTest(SampleTest):

  def test_typical_node(self):
    idx = 0
    # edge 0 has weight 1, edge 1 has weight 2
    idx2features = csr_matrix([
        [1, 0],  # node 0
        [1, 0],  # node 1
        [1, 2],  # node 2
        [0, 2]  # node 3
    ])
    is_edge = False
    actual = [
        SameTypeJaccardSample((idx, 1), idx2features, is_edge),
        SameTypeJaccardSample((idx, 2), idx2features, is_edge),
        SameTypeJaccardSample((idx, 3), idx2features, is_edge)
    ]
    expected = [
        SimilarityRecord(left_node_idx=idx, right_node_idx=1, node_node_prob=1),
        SimilarityRecord(
            left_node_idx=idx, right_node_idx=2, node_node_prob=1 / 3),
        SimilarityRecord(left_node_idx=idx, right_node_idx=3, node_node_prob=0)
    ]
    self.assertSimRecsMatch(actual, expected)

  def test_typical_edge(self):
    idx = 0
    idx2features = csr_matrix([
        [1, 0],  # edge 0
        [1, 0],  # edge 1
        [1, 2],  # edge 2
        [0, 2]  # edge 3
    ])
    is_edge = True
    actual = [
        SameTypeJaccardSample((idx, 1), idx2features, is_edge),
        SameTypeJaccardSample((idx, 2), idx2features, is_edge)
    ]
    expected = [
        SimilarityRecord(left_edge_idx=idx, right_edge_idx=1, edge_edge_prob=1),
        SimilarityRecord(
            left_edge_idx=idx, right_edge_idx=2, edge_edge_prob=1 / 3)
    ]
    self.assertSimRecsMatch(actual, expected)


class SparseMatrixTestCase(unittest.TestCase):

  def assertSparseAlmostEqual(self, actual, expected, tol=1E-5):
    self.assertEqual(actual.shape, expected.shape)
    # >= is quicker than < apparently
    # if any element is >= tol, then np.max returns true
    self.assertFalse(((np.abs(actual - expected) >= tol).max()))


class CentroidFromRowsTest(SparseMatrixTestCase):

  def test_typical(self):
    target2features = csr_matrix([[1, 2, 3], [0, 1, 0], [1, 0, 3]])
    idx2targets = csr_matrix([[1, 0, 1]])
    idx = 0
    actual = csr_matrix(CentroidFromRows(idx, idx2targets, target2features))
    expected = csr_matrix([[1, 1, 3]], dtype=np.float32)
    self.assertSparseAlmostEqual(actual, expected)
