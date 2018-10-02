#!/usr/bin/env python3

import argparse
from pathlib import Path
from hypergraph_embedding import Hypergraph
from hypergraph_embedding.hypergraph_util import ToBlockDiagonal
from hypergraph_embedding.hypergraph_util import CompressRange
from hypergraph_embedding.hypergraph_util import ToCsrMatrix
import cv2
import numpy as np

def ParseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-s",
      "--sort",
      action="store_true",
      help="If set, rearrange hypergraph into block diagonal")
  parser.add_argument(
      "-p",
      "--picture",
      type=str,
      help="Path to resulting img"),
  parser.add_argument(
      "-g",
      "--hypergraph",
      type=str,
      help="Hypergraph proto message"),

  args = parser.parse_args()
  assert args.picture is not None
  assert args.hypergraph is not None
  return args


if __name__ == "__main__":
  args = ParseArgs()
  hypergraph = Hypergraph()
  with open(args.hypergraph, "rb") as proto:
    hypergraph.ParseFromString(proto.read())
  if args.sort:
    hypergraph,_,_ = ToBlockDiagonal(hypergraph)
  else:
    hypergraph,_,_ = CompressRange(hypergraph)
  img = 1 - ToCsrMatrix(hypergraph).astype(np.float32).todense()
  img *= 255

  img = cv2.resize(img, (500, 1000))
  cv2.imwrite(args.picture, img)
