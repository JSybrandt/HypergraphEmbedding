#!/usr/bin/env python3
import argparse
from pathlib import Path
from hypergraph_embedding import HypergraphEmbedding

def ParseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "paths",
      nargs="+",
      help="Must supply two paths to Hypergraph Embedding proto messages.")
  return parser.parse_args()


if __name__ == "__main__":
  args = ParseArgs()
  for path in [Path(p) for p in args.paths]:
    print("Reading", path)
    assert path.is_file()
    original_proto = HypergraphEmbedding()
    try:
      with path.open('rb') as proto:
        original_proto.ParseFromString(proto.read())
    except:
      print("Failed to parse", path)
    new_proto = HypergraphEmbedding()
    new_proto.method_name = original_proto.method_name
    new_proto.dim = original_proto.dim
    for node_idx, embedding in original_proto.node.items():
      new_proto.edge[node_idx].values.extend(embedding.values)
    for edge_idx, embedding in original_proto.edge.items():
      new_proto.node[edge_idx].values.extend(embedding.values)
    assert len(new_proto.node) == len(original_proto.edge)
    assert len(new_proto.edge) == len(original_proto.node)
    with path.open('wb') as proto:
      proto.write(new_proto.SerializeToString())
    print("Wrote", path)
