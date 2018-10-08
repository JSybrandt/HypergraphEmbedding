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
from hypergraph_embedding.hg2v_model import *
from hypergraph_embedding.hg2v_sample import *
from hypergraph_embedding.auto_encoder import *
from hypergraph_embedding.combine_embeddings_util import *

__all__ = [
    # proto
    "Hypergraph",
    "HypergraphEmbedding",
    "EvaluationMetrics",
    "ExperimentalResult",

    # Parsing
    "ParseRawIntoHypergraph",
    "PARSING_OPTIONS",

    # Embedding
    "Embed",
    "EMBEDDING_OPTIONS",
    "DEBUG_SUMMARY_OPTIONS",

    # Experiments
    "EXPERIMENT_OPTIONS",
    "LinkPredictionData",
    "RemoveRandomConnections",
    "SampleMissingConnections",
    "RunLinkPredictionExperiment",
    "LinkPredictionDataToResultProto"
]
