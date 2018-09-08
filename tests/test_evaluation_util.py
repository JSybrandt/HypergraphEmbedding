import unittest
from itertools import product
from hypergraph_embedding import Hypergraph
from hypergraph_embedding import HypergraphEmbedding
from hypergraph_embedding.data_util import *
from hypergraph_embedding.evaluation_util import *
from hypergraph_embedding.embedding import EmbedRandom
from random import random, randint, sample
from scipy.spatial import distance


def isclose(actual, expected, abs_tol=1e-6):
  return abs(actual - expected) < abs_tol


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

  def test_keep_hg_name(self):
    _input = Hypergraph()
    _input.name = "KEEP_ME"
    AddNodeToEdge(_input, 0, 0)
    AddNodeToEdge(_input, 0, 1)
    AddNodeToEdge(_input, 1, 1)
    actual_hg, removed_list = RemoveRandomConnections(_input, 0)
    self.assertEqual(actual_hg.name, "KEEP_ME")

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


class TestEdgeCentroidPrediction(unittest.TestCase):

  def test_typical(self):
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 0)
    AddNodeToEdge(hypergraph, 1, 0)

    embedding = HypergraphEmbedding()
    embedding.dim = 2
    embedding.node[0].values.extend([0, 1])
    embedding.node[1].values.extend([1, 0])
    embedding.node[2].values.extend([0.5, 0.5])

    # default is cosine
    actual = EdgeCentroidPrediction(hypergraph, embedding, [(2, 0)])
    self.assertEqual(set(actual), set([(2, 0)]))

  def test_euclidian(self):
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 0)
    AddNodeToEdge(hypergraph, 1, 0)
    AddNodeToEdge(hypergraph, 2, 0)

    embedding = HypergraphEmbedding()
    embedding.dim = 2
    embedding.node[0].values.extend([0, 1])
    embedding.node[1].values.extend([-1, -1])
    embedding.node[2].values.extend([1, -1])
    embedding.node[3].values.extend([0, 0])

    actual = EdgeCentroidPrediction(
        hypergraph,
        embedding,
        [(3,
          0)],
        distance_function=distance.euclidean)
    # expect that node 3 is inside the triangle
    self.assertEqual(set(actual), set([(3, 0)]))

  def test_multiple_edges(self):
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 0)
    AddNodeToEdge(hypergraph, 1, 0)
    AddNodeToEdge(hypergraph, 0, 1)
    AddNodeToEdge(hypergraph, 1, 1)

    embedding = HypergraphEmbedding()
    embedding.dim = 2
    embedding.node[0].values.extend([0, 1])
    embedding.node[1].values.extend([1, 0])
    embedding.node[2].values.extend([0.5, 0.5])

    actual = EdgeCentroidPrediction(hypergraph, embedding, [(2, 0), (2, 1)])
    self.assertEqual(set(actual), set([(2, 0), (2, 1)]))

  def test_dropped_edge(self):
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 0)
    AddNodeToEdge(hypergraph, 1, 0)
    AddNodeToEdge(hypergraph, 1, 1)  # dropped, only 1

    embedding = HypergraphEmbedding()
    embedding.dim = 2
    embedding.node[0].values.extend([0, 1])
    embedding.node[1].values.extend([1, 0])
    embedding.node[2].values.extend([0.5, 0.5])

    actual = EdgeCentroidPrediction(hypergraph, embedding, [(2, 0), (2, 1)])
    self.assertEqual(set(actual), set([(2, 0)]))

  def test_only_missing_links(self):
    "Given the optional parameter for missing links, don't test everything"
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 0)
    AddNodeToEdge(hypergraph, 1, 0)

    embedding = HypergraphEmbedding()
    embedding.dim = 2
    embedding.node[0].values.extend([0, 1])
    embedding.node[1].values.extend([1, 0])
    embedding.node[2].values.extend([0.5, 0.5])  # detect
    embedding.node[3].values.extend([0.5, 0.5])  # skip

    links = [(2, 0)]

    actual = EdgeCentroidPrediction(hypergraph, embedding, links)
    self.assertEqual(set(actual), set([(2, 0)]))

  def test_missing_edge_removed(self):
    "Tests the case wherein a listed missing edge doesn't appear"
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 0)
    AddNodeToEdge(hypergraph, 1, 0)

    embedding = HypergraphEmbedding()
    embedding.dim = 2
    embedding.node[0].values.extend([0, 1])
    embedding.node[1].values.extend([1, 0])

    # Test each case, none should fail
    links = [(2, 0), (0, 1), (1, 1)]

    actual = EdgeCentroidPrediction(hypergraph, embedding, links)
    self.assertEqual(set(actual), set())


class TestCalculateCommunityPredictionMetrics(unittest.TestCase):

  def test_typical(self):
    predicted = [
        (1, 2), # good prediction
        (2, 1),  # bad prediction
        (2, 3),  # bad prediction
        (2, 4)  # good prediction
        ]
    good_links = [
        (1,
         2),
        (2,
         4),
        (2,
         5)  # missed this one
    ]
    bad_links = [
        (3, 0),  # properly identified!
        (3, 2),  # properly identified!
        (2, 1),  # bad prediction
        (2, 3),  # bad prediction
    ]
    actual = CalculateCommunityPredictionMetrics(
        predicted,
        good_links,
        bad_links)

    # accuracy = # good / all = (2 + 2) / 7
    self.assertTrue(isclose(actual.accuracy, 4 / 7, abs_tol=1e-4))

    # precision = # found / # guessed = 2 / 4
    self.assertTrue(isclose(actual.precision, 2 / 4, abs_tol=1e-4))

    # recall = # found / # to find  = 2 / 3
    self.assertTrue(isclose(actual.recall, 2 / 3, abs_tol=1e-4))

    # f1 = 2 * precision * recall / (precision + recall)
    #    = 2 * (2/4) * (2/3) / (2/4 + 2/3) ~ 0.57
    expected_f1 = 2 * (2 / 4) * (2 / 3) / (2 / 4 + 2 / 3)
    self.assertTrue(isclose(actual.f1, expected_f1, abs_tol=1e-4))

    self.assertEqual(actual.num_true_pos, 2)
    self.assertEqual(actual.num_false_pos, 2)
    self.assertEqual(actual.num_false_neg, 1)
    self.assertEqual(actual.num_true_neg, 2)

  def test_no_predictions(self):
    predicted = []
    good_links = [(1,
                   2)  # missed
                 ]
    bad_links = [(2, 3)]
    actual = CalculateCommunityPredictionMetrics(
        predicted,
        good_links,
        bad_links)
    self.assertTrue(isclose(actual.accuracy, 0.5))
    # precision = # found / # guessed = NAN
    # SHOULD NOT CRASH
    self.assertTrue(not actual.HasField("precision"))
    self.assertTrue(isclose(actual.recall, 0))
    # F1 is also nan at this pont
    self.assertTrue(not actual.HasField("f1"))
    self.assertEqual(actual.num_true_pos, 0)
    self.assertEqual(actual.num_false_pos, 0)
    self.assertEqual(actual.num_false_neg, 1)
    self.assertEqual(actual.num_true_neg, 1)

  def test_no_good(self):
    predicted = [(1, 2)]
    good_links = []
    bad_links = [(1, 2)]
    actual = CalculateCommunityPredictionMetrics(
        predicted,
        good_links,
        bad_links)

    self.assertTrue(actual.accuracy == 0)
    # recall = # found / # num to find = NAN
    # SHOULD NOT CRASH
    self.assertTrue(not actual.HasField("recall"))
    self.assertTrue(isclose(actual.precision, 0))
    # F1 is also nan at this pont
    self.assertTrue(not actual.HasField("f1"))
    self.assertEqual(actual.num_true_pos, 0)
    self.assertEqual(actual.num_false_pos, 1)
    self.assertEqual(actual.num_false_neg, 0)
    self.assertEqual(actual.num_true_neg, 0)


class TestSampleMissingConnections(unittest.TestCase):

  def test_fuzz(self):
    for i in range(100):
      _input = CreateRandomHyperGraph(50, 50, 0.1)
      num_nodes = len(_input.node)
      num_edges = len(_input.edge)
      num_samples = randint(0, 10)
      missing_edges = SampleMissingConnections(_input, num_samples)
      self.assertEqual(len(missing_edges), num_samples)
      for node_idx, edge_idx in missing_edges:
        # legal indices
        self.assertTrue(node_idx in _input.node)
        self.assertTrue(edge_idx in _input.edge)
        # not present edge
        self.assertTrue(edge_idx not in _input.node[node_idx].edges)
        self.assertTrue(node_idx not in _input.edge[edge_idx].nodes)


class TestPersonalizedEdgeClassifiers(unittest.TestCase):

  def DummySetup(self):
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 1)
    AddNodeToEdge(hypergraph, 1, 1)
    AddNodeToEdge(hypergraph, 0, 2)
    AddNodeToEdge(hypergraph, 1, 3)
    AddNodeToEdge(hypergraph, 3, 4)

    embedding = HypergraphEmbedding()
    embedding.node[0].values.extend([0, 0])
    embedding.node[1].values.extend([1, 0])
    embedding.node[3].values.extend([3, 0])
    embedding.edge[1].values.extend([0, 1])
    embedding.edge[2].values.extend([0, 2])
    embedding.edge[3].values.extend([0, 3])
    embedding.edge[4].values.extend([0, 4])

    return hypergraph, embedding

  def test_per_edge_typical(self):
    "Ensures that the edge-wise option provides a dict with classifier"
    "per edge"
    hypergraph, embedding = self.DummySetup()
    actual = GetPersonalizedClassifiers(
        hypergraph,
        embedding,
        per_edge=True,
        run_in_parallel=False)
    self.assertEqual(len(actual), len(hypergraph.edge))
    for edge_idx in hypergraph.edge:
      self.assertTrue(edge_idx in actual)
      self.assertTrue(hasattr(actual[edge_idx], "predict"))

  def test_per_node_typical(self):
    "Ensures that the edge-wise option provides a dict with classifier"
    "per edge"
    hypergraph, embedding = self.DummySetup()
    actual = GetPersonalizedClassifiers(
        hypergraph,
        embedding,
        per_edge=False,
        run_in_parallel=False)
    self.assertEqual(len(actual), len(hypergraph.node))
    for node_idx in hypergraph.node:
      self.assertTrue(node_idx in actual)
      self.assertTrue(hasattr(actual[node_idx], "predict"))

  def test_edge_all_nodes(self):
    "In the case where there exist no negative training examples."
    "We can't crash"
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 1)
    AddNodeToEdge(hypergraph, 1, 1)

    embedding = HypergraphEmbedding()
    embedding.node[0].values.extend([0, 0])
    embedding.node[1].values.extend([1, 0])
    embedding.edge[1].values.extend([0, 1])

    actual = GetPersonalizedClassifiers(
        hypergraph,
        embedding,
        per_edge=True,
        run_in_parallel=False)
    self.assertEqual(len(actual), len(hypergraph.edge))
    for edge_idx in hypergraph.edge:
      self.assertTrue(edge_idx in actual)
      self.assertTrue(hasattr(actual[edge_idx], "predict"))

  def test_edge_no_nodes(self):
    "In this case there is an edge without any nodes, and it doesn't "
    "have an embedding."
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 1)
    AddNodeToEdge(hypergraph, 1, 1)
    hypergraph.edge[2].name = "I'm gonna break it!"

    embedding = HypergraphEmbedding()
    embedding.node[0].values.extend([0, 0])
    embedding.node[1].values.extend([1, 0])
    embedding.edge[1].values.extend([0, 1])

    actual = GetPersonalizedClassifiers(
        hypergraph,
        embedding,
        per_edge=True,
        run_in_parallel=False)
    self.assertEqual(len(actual), len(hypergraph.edge))
    for edge_idx in hypergraph.edge:
      self.assertTrue(edge_idx in actual)
      self.assertTrue(hasattr(actual[edge_idx], "predict"))

  def test_edge_one_node(self):
    "In this case we only have one connection"
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 1)

    embedding = HypergraphEmbedding()
    embedding.node[0].values.extend([0, 0])
    embedding.node[1].values.extend([1, 0])

    actual = GetPersonalizedClassifiers(
        hypergraph,
        embedding,
        per_edge=True,
        run_in_parallel=False)
    self.assertEqual(len(actual), len(hypergraph.edge))
    for edge_idx in hypergraph.edge:
      self.assertTrue(edge_idx in actual)
      self.assertTrue(hasattr(actual[edge_idx], "predict"))

  def test_actually_predicts(self):
    hypergraph, embedding = self.DummySetup()
    actual = GetPersonalizedClassifiers(
        hypergraph,
        embedding,
        per_edge=True,
        run_in_parallel=False)
    # I don't care what it predicts, just don't mess up
    self.assertTrue(actual[1].predict([[0, 1]]) >= 0)

  def test_fuzz(self):
    for i in range(10):
      hypergraph = CreateRandomHyperGraph(100, 100, 0.25)
      embedding = EmbedRandom(hypergraph, 2)
      actual_edge = GetPersonalizedClassifiers(
          hypergraph,
          embedding,
          per_edge=True)
      self.assertEqual(len(actual_edge), len(hypergraph.edge))
      for edge_idx in hypergraph.edge:
        self.assertTrue(edge_idx in actual_edge)
        self.assertTrue(hasattr(actual_edge[edge_idx], "predict"))

      actual_node = GetPersonalizedClassifiers(
          hypergraph,
          embedding,
          per_edge=False)
      self.assertEqual(len(actual_node), len(hypergraph.node))
      for node_idx in hypergraph.node:
        self.assertTrue(node_idx in actual_node)
        self.assertTrue(hasattr(actual_node[node_idx], "predict"))


class TestClassifierPrediction(unittest.TestCase):

  def test_edge_fuzz(self):
    for i in range(10):
      hypergraph = CreateRandomHyperGraph(10, 10, 0.25)
      embedding = EmbedRandom(hypergraph, 2)
      all_pairs = list(product(hypergraph.node, hypergraph.edge))
      potential_links = sample(all_pairs, randint(0, len(all_pairs) - 1))
      predicted_links = ClassifierPrediction(
          hypergraph,
          embedding,
          potential_links,
          per_edge=True)
      # All predicted links must have existed in input
      self.assertEqual(
          len(set(predicted_links).intersection(set(potential_links))),
          len(predicted_links))

  def test_node_fuzz(self):
    for i in range(10):
      hypergraph = CreateRandomHyperGraph(10, 10, 0.25)
      embedding = EmbedRandom(hypergraph, 2)
      all_pairs = list(product(hypergraph.node, hypergraph.edge))
      potential_links = sample(all_pairs, randint(0, len(all_pairs) - 1))
      predicted_links = ClassifierPrediction(
          hypergraph,
          embedding,
          potential_links,
          per_edge=False)
      # All predicted links must have existed in input
      self.assertEqual(
          len(set(predicted_links).intersection(set(potential_links))),
          len(predicted_links))
