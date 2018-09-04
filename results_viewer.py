#!/usr/bin/env python3

# This file investigates the output of ExperimentalResult
# proto messages

import argparse
import sys
from textwrap import dedent
from hypergraph_embedding import *
from pathlib import Path


def ParseArgs():
  parser = argparse.ArgumentParser(
      description="Print results proto information.")
  parser.add_argument(
      "protos",
      type=str,
      help="Path to the results proto buffer",
      nargs="+")
  return parser.parse_args()


def PrintResult(result):
  result_text = dedent(
      """\
  --------------------
  Hypergraph:       {hypergraph}
  Number of Nodes:  {nodes}
  Number of Edges:  {edges}
  --------------------
  Embedding Method: {method}
  Embedding Dim:    {dim}
  --------------------
  {metrics}
  --------------------""")
  print(
      result_text.format(
          hypergraph=result.hypergraph.name,
          nodes=len(result.hypergraph.node),
          edges=len(result.hypergraph.edge),
          method=result.embedding.method_name,
          dim=result.embedding.dim,
          metrics=result.metrics))


if __name__ == "__main__":
  args = ParseArgs()

  proto_paths = [Path(proto) for proto in args.protos]
  for path in proto_paths:
    assert path.is_file()

  for path in proto_paths:
    with path.open('rb') as proto_file:
      result = ExperimentalResult()
      try:
        result.ParseFromString(proto_file.read())
      except:
        print(
            "[ERROR]:{} is not an Experimental Result".format(path),
            file=sys.stderr)
        exit(1)
      PrintResult(result)
