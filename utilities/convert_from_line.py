#!/usr/bin/env python3

import argparse
from pathlib import Path
from hypergraph_embedding import ExperimentalResult
from hypergraph_embedding.data_util import LoadMetadataMaps
import struct

def parse_binary_file(bin_path):
  with open(bin_path, 'rb') as bin_file:
    raw_buffer = bin_file.read()
  header, vector_data = raw_buffer.split(b'\n', 1)
  num_vecs, dim = [int(i) for i in header.decode('ascii').split()]
  in_name=True
  name_buffer=b''
  vec_buffer=b''
  for char in vector_data:
    char = bytes([char])
    if in_name:
      if char == b' ':
        in_name=False
      else:
        name_buffer += char
    else:
      vec_buffer += char
      if len(vec_buffer) == dim*4 + 1:
        yield (int(name_buffer.decode('ascii')),
               struct.unpack(str(dim)+'fx', vec_buffer))
        name_buffer=b''
        vec_buffer=b''
        in_name=True


def ParseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument("line", type=Path)
  parser.add_argument("metadata", type=Path)
  parser.add_argument("reference_exp", type=Path)
  parser.add_argument("out_exp", type=Path)
  return parser.parse_args()

if __name__=="__main__":
  args = ParseArgs()
  print(args)
  assert args.line.is_file()
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
  result_exp.embedding.method_name="LINE"
  load_dim=True
  for idx, vec in parse_binary_file(args.line):
    if load_dim:
      result_exp.embedding.dim = len(vec)
      load_dim = False
    if idx in node_map:
      result_exp.embedding.node[node_map[idx]].values.extend(vec)
    elif idx in edge_map:
      result_exp.embedding.edge[edge_map[idx]].values.extend(vec)
    else:
      print("INVALID METADATA + LINE")
      exit(1)

  print("Loaded", len(result_exp.embedding.node), "node embeddings")
  print("Loaded", len(result_exp.embedding.edge), "edge embeddings")

  with open(args.out_exp, 'wb') as proto:
    proto.write(result_exp.SerializeToString())

