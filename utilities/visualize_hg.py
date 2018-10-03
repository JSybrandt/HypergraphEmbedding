#!/usr/bin/env python3

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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
      "--adj-img",
      type=str,
      help="Path to resulting img. Black pixels represent node-edge adj."),
  parser.add_argument(
      "--distribution-img",
      type=str,
      help="Path to resulting node/edge degree distribution.")
  parser.add_argument(
      "-g", "--hypergraph", type=str, help="Hypergraph proto message"),

  args = parser.parse_args()
  assert args.hypergraph is not None
  return args


def PlotDegreeDistributions(hypergraph, path):
  fig = plt.figure()
  node_dist = fig.add_subplot(211)
  edge_dist = fig.add_subplot(212)

  node_dist.hist([len(n.edges) for _, n in hypergraph.node.items()])
  node_dist.set_title("Node Size Distribution #N={}".format(
      len(hypergraph.node)))
  node_dist.set_yscale("log")

  edge_dist.hist([len(e.nodes) for _, e in hypergraph.edge.items()])
  edge_dist.set_title("Edge Size Distribution #E={}".format(
      len(hypergraph.edge)))
  edge_dist.set_yscale("log")

  fig.suptitle(hypergraph.name)
  fig.tight_layout()
  fig.subplots_adjust(top=0.85)
  fig.savefig(path)


if __name__ == "__main__":
  args = ParseArgs()
  hypergraph = Hypergraph()
  with open(args.hypergraph, "rb") as proto:
    hypergraph.ParseFromString(proto.read())
  if args.sort:
    hypergraph, _, _ = ToBlockDiagonal(hypergraph)
  else:
    hypergraph, _, _ = CompressRange(hypergraph)

  if args.distribution_img is not None:
    PlotDegreeDistributions(hypergraph, args.distribution_img)

  if args.adj_img is not None:
    img = 1 - ToCsrMatrix(hypergraph).astype(np.float32).todense()
    img *= 255

    img = cv2.resize(img, (500, 1000))
    cv2.imwrite(args.adj_img, img)
