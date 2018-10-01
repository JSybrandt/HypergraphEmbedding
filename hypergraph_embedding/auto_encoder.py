################################################################################
# AutoEncoder with helper functions                                            #
################################################################################

import multiprocessing
from multiprocessing import Pool
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
from . import HypergraphEmbedding
from .hypergraph_util import ToCsrMatrix
from .hypergraph_util import ToEdgeCsrMatrix
from .hypergraph_util import CompressRange
import logging
from tqdm import tqdm
import numpy as np
log = logging.getLogger()


def _auto_encoder_sample(row_idx, matrix, num_samples):
  """
  Returns a list of tuples original - perturbed
  Each value is normalized such that it sums to 1.
  """

  sparse_row = matrix[row_idx]
  normalized_row = (sparse_row / sparse_row.nnz).A.flatten()
  perturbed_samples = [normalized_row]
  if sparse_row.nnz > 1:
    nonzero_cols = sparse_row.nonzero()[1]
    num_samples = min(len(nonzero_cols), num_samples)
    for col_idx in np.random.choice(nonzero_cols, num_samples, replace=False):
      perturbed = (sparse_row / (sparse_row.nnz - 1)).A.flatten()
      perturbed[col_idx] = 0
      perturbed_samples.append(perturbed)
  original_samples = [normalized_row] * len(perturbed_samples)
  return original_samples, perturbed_samples


def _get_auto_encoder_embeddings(matrix, dimension, important_indices, epochs,
                                 idx_per_batch, samples_per_idx, disable_pbar):
  log.info("Constructing model")
  sample_size = matrix.shape[1]
  perturbed_input_layer = Input((sample_size,),
                                name="perturbed_input_layer",
                                dtype=np.float32)
  encoding_layer = Dense(
      dimension, name="encoding_layer", activation="relu")(
          perturbed_input_layer)
  original_output_layer = Dense(
      sample_size, name="original_output_layer", activation="softmax")(
          encoding_layer)

  encoding_trainer = Model(
      inputs=[perturbed_input_layer], outputs=[original_output_layer])
  encoding_trainer.compile(
      optimizer="adagrad", loss="kullback_leibler_divergence")

  log.info("Training model")
  indices = np.array(important_indices)
  for epoch in range(epochs):
    log.info("Epoch %i / %i", epoch + 1, epochs)
    for idx_batch in tqdm(
        np.array_split(
            np.random.permutation(indices),
            max(1, int(len(important_indices) / idx_per_batch))),
        disable=disable_pbar):
      originals = []
      perturbed = []
      for idx in idx_batch:
        tmp_o, tmp_p = _auto_encoder_sample(idx, matrix, samples_per_idx)
        originals.extend(tmp_o)
        perturbed.extend(tmp_p)
      originals = np.array(originals, dtype=np.float32)
      perturbed = np.array(perturbed, dtype=np.float32)
      encoding_trainer.train_on_batch(perturbed, originals)

  log.info("Extracting embeddings")
  embedding = Model(inputs=[perturbed_input_layer], outputs=[encoding_layer])
  indices = [idx for idx in important_indices]
  samples = np.array(
      [(matrix[idx] / matrix[idx].nnz).A.flatten() for idx in indices])
  embeddings = embedding.predict(samples)
  return [(idx, embeddings[row, :]) for row, idx in enumerate(indices)]


def EmbedAutoEncoder(hypergraph,
                     dimension,
                     num_samples=100,
                     epochs=5,
                     idx_per_batch=200,
                     run_in_parallel=True,
                     disable_pbar=False):
  workers = multiprocessing.cpu_count() if run_in_parallel else 1

  log.info("Compressing index space")
  compressed, inv_node_map, inv_edge_map = CompressRange(hypergraph)

  embedding = HypergraphEmbedding()
  embedding.dim = dimension
  embedding.method_name = "AutoEncoder"

  log.info("Collecting node samples for auto encoder")
  for node_idx, emb in _get_auto_encoder_embeddings(
      ToCsrMatrix(compressed), dimension, compressed.node, epochs,
      idx_per_batch, num_samples, disable_pbar):
    embedding.node[inv_node_map[node_idx]].values.extend(emb)

  log.info("Collecting edge samples for auto encoder")
  for edge_idx, emb in _get_auto_encoder_embeddings(
      ToEdgeCsrMatrix(compressed), dimension, compressed.edge, epochs,
      idx_per_batch, num_samples, disable_pbar):
    embedding.edge[inv_edge_map[edge_idx]].values.extend(emb)

  return embedding
