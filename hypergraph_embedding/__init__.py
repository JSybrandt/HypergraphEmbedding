"""
This package is responsible for embedding hypergraphs
Dependencies:
 - Protocol Buffers
 - Numpy
 - SKLearn
"""
from hypergraph_embedding.hypergraph_pb2 import *
from hypergraph_embedding.hypergraph_util import *
from hypergraph_embedding.data_util import *
from hypergraph_embedding.embedding import *
from hypergraph_embedding.evaluation_util import *

__all__ = [
    # proto
    "Hypergraph",
    "HypergraphEmbedding",
    "EvaluationMetrics",

    # Parsing
    "AMinerToHypergraph",
    "SnapCommunityToHypergraph",

    # Embedding
    "EmbedSvd",
    "EmbedRandom",

    # Experiments
    "RunLinkPrediction"
]
