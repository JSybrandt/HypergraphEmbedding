#!/usr/bin/env python3

# Writes plaintext edge list to std_out

import argparse
from pathlib import Path
from hypergraph_embedding import ExperimentalResult
from hypergraph_embedding.data_util import SaveEdgeList
from hypergraph_embedding.data_util import LoadEdgeList

def ParseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument("ref_result", type=Path)
  parser.add_argument("out_data", type=Path)
  parser.add_argument("out_meta", type=Path)
  parser.add_argument("--weighted", action="store_true")
  parser.add_argument("--only-one-side", action="store_true")
  return parser.parse_args()

def test_equal_hypergraphs(a, b):
  assert len(a.node) == len(b.node)
  assert len(a.edge) == len(b.edge)
  for a_idx, a_node in a.node.items():
    assert a_idx in b.node
    b_node = b.node[a_idx]
    assert set(a_node.edges) == set(b_node.edges)
  for a_idx, a_edge in a.edge.items():
    assert a_idx in b.edge
    b_edge = b.edge[a_idx]
    assert set(a_edge.nodes) == set(b_edge.nodes)

if __name__ == "__main__":
  args = ParseArgs()
  print(args)
  assert args.ref_result.is_file()
  assert not args.out_data.is_file()
  assert not args.out_meta.is_file()
  result = ExperimentalResult()

  with open(args.ref_result, 'rb') as file:
    result.ParseFromString(file.read())

  SaveEdgeList(result.hypergraph, args.out_data, args.out_meta, args.weighted, args.only_one_side)

  # TEST WRITE CORRECT
  # recovered = LoadEdgeList(args.out_data, args.out_meta)
  # test_equal_hypergraphs(result.hypergraph, recovered)
