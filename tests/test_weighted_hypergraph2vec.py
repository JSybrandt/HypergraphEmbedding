import unittest
from hypergraph_embedding import Hypergraph
from hypergraph_embedding import HypergraphEmbedding
from hypergraph_embedding.hypergraph_util import *
from hypergraph_embedding.weighted_hypergraph2vec import *
from hypergraph_embedding.embedding import *


class ComputeAlgebraicRadiusTest(unittest.TestCase):

  def test_one_pair(self):
    "Given an embedding, find the distance to each node/edge first "
    "order neighbor"
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 3, 7)
    embedding = HypergraphEmbedding()
    embedding.node[3].values.extend([0])
    embedding.edge[7].values.extend([1])

    node2radius, edge2radius = ComputeAlgebraicRadius(
        hypergraph, embedding, disable_pbar=True)
    self.assertEqual(node2radius, {3: 1})
    self.assertEqual(edge2radius, {7: 1})

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

    node2radius, edge2radius = ComputeAlgebraicRadius(
        hypergraph, embedding, disable_pbar=True)
    self.assertEqual(node2radius, {3: 1, 5: 2})
    self.assertEqual(edge2radius, {7: 2})

  def test_two_d(self):
    "Given 2d embeddings, we should be using l2 norm"

    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 10, 20)
    embedding = HypergraphEmbedding()
    embedding.node[10].values.extend([0, 0])
    embedding.edge[20].values.extend([3, 4])

    # Note the difference between 10 and 20 makes a 3-4-5 triangle
    node2radius, edge2radius = ComputeAlgebraicRadius(
        hypergraph, embedding, disable_pbar=True)
    self.assertEqual(node2radius, {10: 5})
    self.assertEqual(edge2radius, {20: 5})

  def test_no_link_zero(self):
    "If we have a node without connections, treat its radius as 0"
    hypergraph = Hypergraph()
    hypergraph.node[1].name = "I shouldn't mess stuff up"
    hypergraph.edge[2].name = "I shouldn't mess stuff up"

    AddNodeToEdge(hypergraph, 0, 0)

    embedding = HypergraphEmbedding()
    embedding.node[1].values.extend([1, 2])
    embedding.edge[2].values.extend([3, 4])
    embedding.node[0].values.extend([0, 0])
    embedding.edge[0].values.extend([3, 4])

    node2radius, edge2radius = ComputeAlgebraicRadius(
        hypergraph, embedding, disable_pbar=True)
    self.assertEqual(node2radius, {0: 5, 1: 0})
    self.assertEqual(edge2radius, {0: 5, 2: 0})

  def test_fuzz(self):
    "Needs to make its own embedding if I don't supply one"
    for i in range(10):
      dim = 2
      hypergraph = CreateRandomHyperGraph(10, 10, 0.1)
      embedding = EmbedRandom(hypergraph, 2)
      # make sure it doesn't break
      node2rad, edge2rad = ComputeAlgebraicRadius(
          hypergraph, embedding, disable_pbar=True)
      for node_idx in hypergraph.node:
        self.assertTrue(node_idx in node2rad)
        self.assertTrue(node2rad[node_idx] >= 0)
      for edge_idx in hypergraph.edge:
        self.assertTrue(edge_idx in edge2rad)
        self.assertTrue(edge2rad[edge_idx] >= 0)
