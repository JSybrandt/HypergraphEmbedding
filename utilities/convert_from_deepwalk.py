#!/usr/bin/env python3

import argparse
from pathlib import Path
from hypergraph_embedding import ExperimentalResult
from hypergraph_embedding.data_util import LoadMetadataMaps

def ParseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument("deepwalk", type=Path)
  parser.add_argument("metadata", type=Path)
  parser.add_argument("reference_exp", type=Path)
  parser.add_argument("out_exp", type=Path)
  return parser.parse_args()

if __name__=="__main__":
  args = ParseArgs()
  print(args)
  assert args.deepwalk.is_file()
  assert args.metadata.is_file()
  assert args.reference_exp.is_file()
  assert not args.out_exp.is_file()
  assert args.out_exp.parent.is_dir()
  node_map, edge_map = LoadMetadataMaps(args.metadata)

  result_exp = ExperimentalResult()
  # Copy all
  with open(args.reference_exp, 'rb') as proto:
    result_exp.ParseFromString(proto.read())
  # Clear embedding
  result_exp.embedding.Clear()

  # Load Emb
  result_exp.embedding.method_name="deepwalk"
  load_dim=True
  with open(args.deepwalk) as file:
    next(file) # skip header line
    for line in file:
      tokens = line.split()
      idx = int(tokens[0])
      vec = [float(f) for f in tokens[1:]]
      if load_dim:
        result_exp.embedding.dim = len(vec)
        load_dim = False
      if idx in node_map:
        result_exp.embedding.node[node_map[idx]].values.extend(vec)
      elif idx in edge_map:
        result_exp.embedding.edge[edge_map[idx]].values.extend(vec)
      else:
        print("INVALID METADATA + DEEPWALK")
        exit(1)

  print("Loaded", len(result_exp.embedding.node), "node embeddings")
  print("Loaded", len(result_exp.embedding.edge), "edge embeddings")

  with open(args.out_exp, 'wb') as proto:
    proto.write(result_exp.SerializeToString())

