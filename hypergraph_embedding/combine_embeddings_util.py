import logging
import numpy as np
from . import HypergraphEmbedding
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Concatenate
from .evaluation_util import SampleMissingConnections

log = logging.getLogger()


def _concatenate_embeddings(indices, half_embeddings):
  idx2emb = {}
  for idx in indices:
    vec = []
    for emb in half_embeddings:
      vec.extend(emb[idx].values)
    idx2emb[idx] = np.array(vec, dtype=np.float32)
  assert len(indices) == len(idx2emb)
  return idx2emb

def _sample_hypergraph(hypergraph, node2emb, edge2emb):
  log.info("Collecting positive samples")
  node_samples = []
  edge_samples = []
  labels = []
  for node_idx, node in hypergraph.node.items():
    for edge_idx in node.edges:
      node_samples.append(node2emb[node_idx])
      edge_samples.append(edge2emb[edge_idx])
      labels.append(1)
  num_pos_samples = len(labels)

  log.info("Collecting negative samples")
  for node_idx, edge_idx in SampleMissingConnections(hypergraph,
                                                     5*num_pos_samples):
    node_samples.append(node2emb[node_idx])
    edge_samples.append(edge2emb[edge_idx])
    labels.append(0)

  return node_samples, edge_samples, labels

def _interpret_model(indices, idx2cat_emb, predictor, half_emb):
  vecs = np.array([idx2cat_emb[idx] for idx in indices])
  embeddings = predictor.predict(vecs)
  for row_idx, idx, in enumerate(indices):
    half_emb[idx].values.extend(embeddings[row_idx, :])


def CombineEmbeddingsViaNodeEdgeClassifier(hypergraph, embeddings,
                                                desired_dim, disable_pbar):
  assert desired_dim > 0

  log.info("Concatenating node embedding")
  node2cat_emb = _concatenate_embeddings(hypergraph.node, [emb.node for emb in embeddings])
  log.info("Concatenating edge embedding")
  edge2cat_emb = _concatenate_embeddings(hypergraph.edge, [emb.edge for emb in embeddings])
  node_samples, edge_samples, labels = _sample_hypergraph(hypergraph, node2cat_emb, edge2cat_emb)

  ## Constructing Model
  log.info("Constucting model")
  input_size = sum([emb.dim for emb in embeddings])

  concatenated_node = Input(
      (input_size,), dtype=np.float32, name="ConcatinatedNode")
  concatenated_edge = Input(
      (input_size,), dtype=np.float32, name="ConcatinatedEdge")

  joint_node = Dense(
      desired_dim, dtype=np.float32, activation="relu", name="JointNode")(
          concatenated_node)
  joint_edge = Dense(
      desired_dim, dtype=np.float32, activation="relu", name="JointEdge")(
          concatenated_edge)

  merged = Concatenate(axis=1)([joint_node, joint_edge])
  out = Dense(1, activation="sigmoid")(merged)

  emb_trainer = Model(
      inputs=[concatenated_node, concatenated_edge], outputs=[out])
  emb_trainer.compile(optimizer="adagrad", loss="mean_squared_error")

  stopper = EarlyStopping(monitor="loss")

  log.info("Training model")
  emb_trainer.fit(
      [node_samples, edge_samples],
      labels,
      batch_size=256,
      epochs=50,
      callbacks=[stopper],  # Stop if actually no improvement
      verbose=0 if disable_pbar else 1)

  ## Interpreting Model

  embedding = HypergraphEmbedding()
  embedding.dim = desired_dim

  log.info("Interpreting model for compressed node embeddings")
  node_predictor = Model(inputs=[concatenated_node], outputs=[joint_node])
  _interpret_model(hypergraph.node, node2cat_emb, node_predictor, embedding.node)

  log.info("Interpreting model for compressed edge embeddings")
  edge_predictor = Model(inputs=[concatenated_edge], outputs=[joint_edge])
  _interpret_model(hypergraph.edge, edge2cat_emb, edge_predictor, embedding.edge)

  return embedding
