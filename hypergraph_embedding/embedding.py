# This file contains embedding objects to project hypergraph nodes and/or edges
# into a dense vector space.

import scipy as sp
from . import ToCsrMatrix
from collections.abc import Mapping


class Embedding(Mapping):
  "The Embedding ABC describes the embedding interface"

  def __init__(self, dim):
    "Initializes an embedding object that projects a hypergraph into "
    "`dim` dimensional space."
    self._dim = dim
    self._embedding_dict = {}

  def embed(self, hypergraph):
    pass

  def __getitem__(self, key):
    "Returns an embedding. Embeddings are read-only"
    assert key in self._embedding_dict
    return self._embedding_dict[key]

  def __len__(self):
    return len(self._embedding_dict)

  def __iter__(self):
    return iter(self._embedding_dict)

  def items(self):
    return self._embedding_dict.items()


class SvdEmbedding(Embedding):

  def __init__(self, dim):
    super(SvdEmbedding, self).__init__(dim)

  def embed(self, hypergraph):
    # SVD cannot embed a to a higher dimension than matrix order
    assert self._dim < len(hypergraph.node)
    assert self._dim < len(hypergraph.edge)

    matrix = ToCsrMatrix(hypergraph)
    U, _, _ = embedding_data = sp.sparse.linalg.svds(
        matrix, self._dim, return_singular_vectors='u')
    for node_idx in hypergraph.node:
      self._embedding_dict[node_idx] = U[node_idx, :]
