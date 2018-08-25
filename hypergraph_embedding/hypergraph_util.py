# This file contains functions for generating and manipulating the Hypergraph proto message.

from . import hypergraph_pb2 as pb
import numpy as np
import scipy as sp
from random import random

def AddNodeToEdge(hypergraph, node_id, edge_id):
    """
    Modifies hypergraph by setting a connection from given node to given edge.
    """
    if edge_id not in hypergraph.node[node_id].edges:
        hypergraph.node[node_id].edges.append(edge_id)
    if node_id not in hypergraph.edge[edge_id].nodes:
        hypergraph.edge[edge_id].nodes.append(node_id)
    return hypergraph

def CreateRandomHyperGraph(num_nodes, num_edges, probability):
    """
    Creates a graph of `num_nodes` and `num_edges` where `probability` is the chance that node i belongs to edge j.
    """
    assert probability <= 1
    assert probability >= 0
    assert num_edges >= 0
    assert num_nodes >= 0
    result = pb.Hypergraph()
    for i in range(num_nodes):
        for j in range(num_edges):
            if random() < probability:
                AddNodeToEdge(result, i, j)
    return result

def FromSparseMatrix(csr_matrix):
    """
    Creates a hypergraph object from the provided sparse matrix. Each row represents a node, each column represents an edge. A 1 in row i and column j represents that node i belongs to edge j.
    """

def ToSparseMatrix(hypergraph):
    """
    ToSparseMatrix accepts a hypergraph proto message and converts it to a Compressed Sparse Row matrix via scipy. Each row represents a node, each column represents an edge. A 1 in row i and column j represents that node i belongs to edge j.
    """
    return 1


