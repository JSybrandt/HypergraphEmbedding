#/usr/bin/env python3

import unittest
from hypergraph_embedding.hg2v_sample import SimilarityRecord
from hypergraph_embedding.hg2v_sample import SamplesToModelInput


class SamplesToModelInputTest(unittest.TestCase):

  def test_node_sim_unweighted(self):
    _input = SimilarityRecord(
        left_node_idx=0,
        right_node_idx=1,
        node_node_prob=0.5)
    actual = SamplesToModelInput([_input], num_neighbors=2, weighted=False)
    # note, the node indices should be incremented
    expected = ([[1],          # left_node_idx
                 [0],          # left_edge_idx
                 [2],          # right_node_idx
                 [0],          # right_edge_idx
                 [0], [0],     # neighbor_node_indices
                 [0], [0]],    # neighbor_edge_indices
                 [[0.5],       # node_node_prob
                  [0],         # edge_edge_prob
                  [0]])        # node_edge_prob
    self.assertEqual(actual, expected)

  def test_edge_sim_unweighted(self):
    _input = SimilarityRecord(
        left_edge_idx=0,
        right_edge_idx=1,
        edge_edge_prob=0.5)
    actual = SamplesToModelInput([_input], num_neighbors=2, weighted=False)
    # note, the node indices should be incremented
    expected = ([[0],          # left_node_idx
                 [1],          # left_edge_idx
                 [0],          # right_node_idx
                 [2],          # right_edge_idx
                 [0], [0],     # neighbor_node_indices
                 [0], [0]],    # neighbor_edge_indices
                 [[0],         # node_node_prob
                  [0.5],       # edge_edge_prob
                  [0]])        # node_edge_prob
    self.assertEqual(actual, expected)

  def test_edge_node_sim_unweighted(self):
    _input = SimilarityRecord(
        left_node_idx=0,
        right_edge_idx=1,
        neighbor_node_indices=[2],
        neighbor_edge_indices=[3,
                               4],
        node_edge_prob=0.5)
    actual = SamplesToModelInput([_input], num_neighbors=2, weighted=False)
    # note, the node indices should be incremented
    expected = ([[1],          # left_node_idx
                 [0],          # left_edge_idx
                 [0],          # right_node_idx
                 [2],          # right_edge_idx
                 [3], [0],     # neighbor_node_indices
                 [4], [5]],    # neighbor_edge_indices
                 [[0],         # node_node_prob
                  [0],         # edge_edge_prob
                  [0.5]])      # node_edge_prob
    self.assertEqual(actual, expected)

  def test_node_sim_weighted(self):
    _input = SimilarityRecord(
        left_node_idx=0,
        right_node_idx=1,
        left_weight=0.3,
        right_weight=0.6,
        node_node_prob=0.5)
    actual = SamplesToModelInput([_input], num_neighbors=2)
    # note, the node indices should be incremented
    expected = ([[1],          # left_node_idx
                 [0],          # left_edge_idx
                 [2],          # right_node_idx
                 [0],          # right_edge_idx
                 [0.3],        # left_weight
                 [0.6],        # right_weight
                 [0], [0],     # neighbor_node_indices
                 [0], [0],     # neighbor_node_weights
                 [0], [0],     # neighbor_edge_indices
                 [0], [0]],    # neighbor_edge_weights
                 [[0.5],       # node_node_prob
                  [0],         # edge_edge_prob
                  [0]])        # node_edge_prob
    self.assertEqual(actual, expected)

  def test_edge_sim_weighted(self):
    _input = SimilarityRecord(
        left_edge_idx=0,
        right_edge_idx=1,
        left_weight=0.3,
        right_weight=0.6,
        neighbor_node_indices=[2],
        neighbor_node_weights=[0.25],
        neighbor_edge_indices=[3,
                               4],
        neighbor_edge_weights=[0.5,
                               0.75],
        node_edge_prob=0.5)
    actual = SamplesToModelInput([_input], num_neighbors=2)
    # note, the node indices should be incremented
    expected = ([[0],           # left_node_idx
                 [1],           # left_edge_idx
                 [0],           # right_node_idx
                 [2],           # right_edge_idx
                 [0.3],         # left_weight
                 [0.6],         # right_weight
                 [3], [0],      # neighbor_node_indices
                 [0.25], [0],   # neighbor_node_weights
                 [4], [5],      # neighbor_edge_indices
                 [0.5], [.75]], # neighbor_edge_weights
                 [[0],          # node_node_prob
                  [0],          # edge_edge_prob
                  [0.5]])       # node_edge_prob
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
    expected = ([[0],          # left_node_idx
                 [1],          # left_edge_idx
                 [0],          # right_node_idx
                 [2],          # right_edge_idx
                 [0.3],        # left_weight
                 [0.6],        # right_weight
                 [0], [0],     # neighbor_node_indices
                 [0], [0],     # neighbor_node_weights
                 [0], [0],     # neighbor_edge_indices
                 [0], [0]],    # neighbor_edge_weights
                 [[0],         # node_node_prob
                  [0.5],       # edge_edge_prob
                  [0]])        # node_edge_prob
    self.assertEqual(actual, expected)


if __name__ == "__main__":
  unittest.main()
