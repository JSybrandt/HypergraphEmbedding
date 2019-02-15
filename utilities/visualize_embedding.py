import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
from hypergraph_embedding import HypergraphEmbedding
from pathlib import Path
import numpy as np
from scipy.linalg import svd
from random import sample


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("embedding", type=Path)
  parser.add_argument("outpath", type=Path)
  return parser.parse_args()


def PrintPicture(embedding, path, num_samples):
  fig, (node_ax, edge_ax) = plt.subplits(1, 2, figsize=(10,5))

  def project_2d_samples(idx2emb):
    raw = np.zeros((num_samples, embedding.dim), dtype=np.float16)
    for row, idx in enumerate(
        sample(idx2emb.keys(), min(num_samples, len(idx2emb)))):
      raw[row, :] = idx2emb[idx].values
    U, _, _ = svd(raw, full_matrices=0, overwrite_a=True)
    return (U[:, 0], U[:, 1])

  x, y = project_2d_samples(embedding.node)
  node_ax.scatter(x, y)
  node_ax.set_title("Nodes")

  x, y = project_2d_samples(embedding.edge)
  edge_ax.scatter(x, y)
  edge_ax.set_title("Edges")

  plt.tight_layout()
  fig.suptitle("{}:{}".format(embedding.method_name,
                              embedding.dim))
  fig.subplots_adjust(top=0.9)
  fig.savefig(path)

if __name__ == "__main__":
  args = parse_args()
  assert args.embedding.is_file()
  assert not args.outpath.exists()

  emb = HypergraphEmbedding()
  with open(args.embedding, "rb") as proto:
    emb.ParseFromString(proto.read())

  num_samples = min(min(len(emb.node), len(emb.edge)), 10000)

  PrintPicture(emb, str(args.outpath), num_samples)

