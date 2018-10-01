#/usr/bin/env python3

import unittest
from hypergraph_embedding.hg2v_model import KerasModelToEmbedding
from hypergraph_embedding.hg2v_model import BooleanModel
from hypergraph_embedding.hg2v_model import UnweightedFloatModel
from hypergraph_embedding import Hypergraph
from hypergraph_embedding import HypergraphEmbedding
from hypergraph_embedding.hypergraph_util import AddNodeToEdge


class ModelsToEmbedding(unittest.TestCase):

  def test_boolean_model_to_emb(self):
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 0)
    AddNodeToEdge(hypergraph, 2, 2)

    node_map = {
        0: 0,
        2: 1,
    }

    edge_map = {
        0: 0,
        2: 1,
    }

    # should start random?
    dimension = 5
    model = BooleanModel(hypergraph, dimension, 2)
    emb = KerasModelToEmbedding(hypergraph, model, node_map, edge_map)

    self.assertEqual(emb.dim, dimension)
    for node_idx in hypergraph.node:
      self.assertTrue(node_map[node_idx] in emb.node)
      self.assertEqual(len(emb.node[node_map[node_idx]].values), dimension)
    for edge_idx in hypergraph.edge:
      self.assertTrue(edge_map[edge_idx] in emb.edge)
      self.assertEqual(len(emb.edge[edge_map[edge_idx]].values), dimension)

  def test_unweighted_float_model_to_emb(self):
    hypergraph = Hypergraph()
    AddNodeToEdge(hypergraph, 0, 0)
    AddNodeToEdge(hypergraph, 2, 2)

    node_map = {
        0: 0,
        2: 1,
    }

    edge_map = {
        0: 0,
        2: 1,
    }

    # should start random?
    dimension = 5
    model = UnweightedFloatModel(hypergraph, dimension, 2)
    emb = KerasModelToEmbedding(hypergraph, model, node_map, edge_map)

    self.assertEqual(emb.dim, dimension)
    for node_idx in hypergraph.node:
      self.assertTrue(node_map[node_idx] in emb.node)
      self.assertEqual(len(emb.node[node_map[node_idx]].values), dimension)
    for edge_idx in hypergraph.edge:
      self.assertTrue(edge_map[edge_idx] in emb.edge)
      self.assertEqual(len(emb.edge[edge_map[edge_idx]].values), dimension)


if __name__ == "__main__":
  unittest.main()
