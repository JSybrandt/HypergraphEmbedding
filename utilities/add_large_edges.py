#!/usr/bin/env python3

import argparse
from pathlib import Path
from hypergraph_embedding import Hypergraph
from hypergraph_embedding.hypergraph_util import AddNodeToEdge
from random import random

def add_x_edges(num_edges, hypergraph):
  edge_prob = 1.0 / float(num_edges)
  for _ in range(num_edges):
    print("Adding edges with prob", edge_prob)
    count = 0
    new_edge_idx = max(hypergraph.edge) + 1
    for node_idx in hypergraph.node:
      if random() < edge_prob:
        AddNodeToEdge(hypergraph, node_idx, new_edge_idx)
        count += 1
    print("Actual Addition Probability:", count / len(hypergraph.node))

def ParseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "input",
      type=str,
      help="Path to the original hypergraph")
  parser.add_argument(
      "output",
      type=str,
      help="Path to the output hypergraph")
  return parser.parse_args()

if __name__=="__main__":
  args = ParseArgs()

  in_path = Path(args.input)
  assert in_path.is_file()
  out_path = Path(args.output)
  assert not out_path.exists()
  assert out_path.parent.is_dir()

  hypergrah = Hypergraph()
  with in_path.open('rb') as proto:
    hypergrah.ParseFromString(proto.read())

  add_x_edges(5, hypergrah)

  with out_path.open('wb') as proto:
    proto.write(hypergrah.SerializeToString())
