"""
This package is responsible for embedding hypergraphs
Dependencies:
 - Protocol Buffers
 - Numpy
 - SKLearn
"""
from hypergraph_embedding.hypergraph_pb2 import Hypergraph
from hypergraph_embedding.hypergraph_util import ToSparseMatrix

__all__ = [
  "Hypergraph",
  "ToSparseMatrix"
]
