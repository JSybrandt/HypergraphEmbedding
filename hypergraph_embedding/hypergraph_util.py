# This file contains functions for generating and manipulating the Hypergraph proto message.

from . import hypergraph_pb2
import numpy as np
import scipy as sp

def ToSparseMatrix(hypergraph):
    """
    ToSparseMatrix accepts a hypergraph proto message and converts it to a Compressed Sparse Row matrix via scipy. Each row represents a node, each column represents an edge. A 1 in row i and column j reprents node i belongs to edge j.
    """
    return 1


