from . import HypergraphEmbedding
import logging
import numpy as np
import keras
from keras.callbacks import TensorBoard
from keras.layers import Activation
from keras.layers import Average
from keras.layers import Dense
from keras.layers import Dot
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Multiply
from keras.models import Model

log = logging.getLogger()


def KerasModelToEmbedding(
    hypergraph,
    model,
    node_layer_name="node_embedding",
    edge_layer_name="edge_embedding"):
  "Given a trained model, extract embedding weights into proto"
  node_weights = model.get_layer(node_layer_name).get_weights()[0]
  edge_weights = model.get_layer(edge_layer_name).get_weights()[0]

  embedding = HypergraphEmbedding()
  embedding.dim = len(node_weights[0])

  for node_idx in hypergraph.node:
    embedding.node[node_idx].values.extend(node_weights[node_idx + 1])
  for edge_idx in hypergraph.edge:
    embedding.edge[edge_idx].values.extend(edge_weights[edge_idx + 1])
  return embedding


def BooleanModel(hypergraph, dimension, num_neighbors):
  """
  This function produces a keras model designed to train embeddings based on
  the output of BooleanSamples.
  """
  log.info("Constructing Keras Model")

  max_node_idx = max([i for i in hypergraph.node])
  max_edge_idx = max([i for i in hypergraph.edge])

  left_node_idx = Input((1,), name="left_node_idx", dtype=np.int32)
  left_edge_idx = Input((1,), name="left_edge_idx", dtype=np.int32)
  right_node_idx = Input((1,), name="right_node_idx", dtype=np.int32)
  right_edge_idx = Input((1,), name="right_edge_idx", dtype=np.int32)
  neighbor_edge_indices = [
      Input((1,
            ),
            dtype=np.int32,
            name="edges_containing_node_{}".format(i))
      for i in range(num_neighbors)
  ]
  neighbor_node_indices = [
      Input((1,
            ),
            dtype=np.int32,
            name="nodes_in_edge_{}".format(i)) for i in range(num_neighbors)
  ]

  node_emb = Embedding(
      input_dim=max_node_idx + 2,
      output_dim=dimension,
      input_length=1,
      name="node_embedding")
  edge_emb = Embedding(
      input_dim=max_edge_idx + 2,
      output_dim=dimension,
      input_length=1,
      name="edge_embedding")

  # Pad each input, get embeddings
  left_node_vec = Flatten()(node_emb(left_node_idx))
  left_edge_vec = Flatten()(edge_emb(left_edge_idx))
  right_node_vec = Flatten()(node_emb(right_node_idx))
  right_edge_vec = Flatten()(edge_emb(right_edge_idx))

  # calculate expected probabilities
  node_node_prob = Activation(
      "sigmoid",
      name="node_node_prob")(
          Dot(1)([left_node_vec,
                  right_node_vec]))
  edge_edge_prob = Activation(
      "sigmoid",
      name="edge_edge_prob")(
          Dot(1)([left_edge_vec,
                  right_edge_vec]))

  # Get neighborhood embeddings
  sig = Activation("sigmoid")
  nodes_dot_sigs = [
      sig(Dot(1)([Flatten()(node_emb(node)),
                  left_node_vec])) for node in neighbor_node_indices
  ]
  edges_dot_sigs = [
      sig(Dot(1)([Flatten()(edge_emb(edge)),
                  right_edge_vec])) for edge in neighbor_edge_indices
  ]

  node_sig_avg = Average()(nodes_dot_sigs)
  edge_sig_avg = Average()(edges_dot_sigs)
  node_edge_prob = Multiply(name="node_edge_prob")([node_sig_avg, edge_sig_avg])
  model = Model(
      inputs=[left_node_idx,
              left_edge_idx,
              right_node_idx,
              right_edge_idx] \
             + neighbor_node_indices \
             + neighbor_edge_indices,
      outputs=[node_node_prob,
               edge_edge_prob,
               node_edge_prob])
  model.compile(optimizer="adagrad", loss="kullback_leibler_divergence")
  return model


def UnweightedFloatModel(hypergraph, dimension, num_neighbors):
  """
  This function produces a keras model designed to train embeddings based on
  the output of AdjJaccardSamples.
  """
  log.info("Constructing Keras Model")

  max_node_idx = max([i for i in hypergraph.node])
  max_edge_idx = max([i for i in hypergraph.edge])

  left_node_idx = Input((1,), name="left_node_idx", dtype=np.int32)
  left_edge_idx = Input((1,), name="left_edge_idx", dtype=np.int32)
  right_node_idx = Input((1,), name="right_node_idx", dtype=np.int32)
  right_edge_idx = Input((1,), name="right_edge_idx", dtype=np.int32)
  neighbor_edge_indices = [
      Input((1,
            ),
            dtype=np.int32,
            name="edges_containing_node_{}".format(i))
      for i in range(num_neighbors)
  ]
  neighbor_node_indices = [
      Input((1,
            ),
            dtype=np.int32,
            name="nodes_in_edge_{}".format(i)) for i in range(num_neighbors)
  ]

  node_emb = Embedding(
      input_dim=max_node_idx + 2,
      output_dim=dimension,
      input_length=1,
      name="node_embedding")
  edge_emb = Embedding(
      input_dim=max_edge_idx + 2,
      output_dim=dimension,
      input_length=1,
      name="edge_embedding")

  # Pad each input, get embeddings
  left_node_vec = Flatten()(node_emb(left_node_idx))
  left_edge_vec = Flatten()(edge_emb(left_edge_idx))
  right_node_vec = Flatten()(node_emb(right_node_idx))
  right_edge_vec = Flatten()(edge_emb(right_edge_idx))

  # calculate expected probabilities
  node_node_prob = Activation(
      "relu",
      name="node_node_prob")(
          Dot(1)([left_node_vec,
                  right_node_vec]))
  edge_edge_prob = Activation(
      "relu",
      name="edge_edge_prob")(
          Dot(1)([left_edge_vec,
                  right_edge_vec]))

  # Get neighborhood embeddings
  sig = Activation("relu")
  nodes_dot_sigs = [
      sig(Dot(1)([Flatten()(node_emb(node)),
                  left_node_vec])) for node in neighbor_node_indices
  ]
  edges_dot_sigs = [
      sig(Dot(1)([Flatten()(edge_emb(edge)),
                  right_edge_vec])) for edge in neighbor_edge_indices
  ]

  node_sig_avg = Average()(nodes_dot_sigs)
  edge_sig_avg = Average()(edges_dot_sigs)
  node_edge_prob = Multiply(name="node_edge_prob")([node_sig_avg, edge_sig_avg])
  model = Model(
      inputs=[left_node_idx,
              left_edge_idx,
              right_node_idx,
              right_edge_idx] \
             + neighbor_node_indices \
             + neighbor_edge_indices,
      outputs=[node_node_prob,
               edge_edge_prob,
               node_edge_prob])
  model.compile(optimizer="adagrad", loss="mean_squared_error")
  return model
