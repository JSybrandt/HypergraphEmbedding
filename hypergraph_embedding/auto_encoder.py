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
import logging
from tqdm import tqdm
import numpy as np
log = logging.getLogger()

# Used to coordinate parallel processes
_shared_info = {}


def _init_auto_encoder_sample(matrix):
  _shared_info.clear()
  _shared_info["matrix"] = matrix


def _auto_encoder_sample(row_idx, matrix=None):
  """
  Returns a list of tuples original - perturbed
  Each value is normalized such that it sums to 1.
  """
  if matrix is None:
    matrix = _shared_info["matrix"]

  sparse_row = matrix[row_idx]
  normalized_row = (sparse_row / sparse_row.nnz).A.flatten()

  samples = [(normalized_row, normalized_row)]
  if sparse_row.nnz > 1:
    for col_idx in sparse_row.nonzero()[1]:
      perturbed = (sparse_row / (sparse_row.nnz - 1)).A.flatten()
      perturbed[col_idx] = 0
      samples.append((normalized_row, perturbed))
  return samples


def _to_samples(indices, matrix):
  return {idx: (matrix[idx] / matrix[idx].nnz).A.flatten() for idx in indices}


def _get_auto_encoder_embeddings(samples, dimension, idx2sample, disable_pbar):
  log.info("Converting samples to input arrays")
  #samples is original, perturbed
  originals = np.array([o for o, _ in samples])
  perturbed = np.array([p for _, p in samples])
  sample_size = originals.shape[1]
  assert originals.shape == perturbed.shape

  log.info("Constructing model")
  perturbed_input_layer = Input((sample_size,
                                ),
                                name="perturbed_input_layer",
                                dtype=np.float32)
  encoding_layer = Dense(
      dimension,
      name="encoding_layer",
      activation="softmax")(
          perturbed_input_layer)
  original_output_layer = Dense(
      sample_size,
      name="original_output_layer",
      activation="softmax")(
          encoding_layer)

  encoding_trainer = Model(
      inputs=[perturbed_input_layer],
      outputs=[original_output_layer])
  encoding_trainer.compile(
      optimizer="adagrad",
      loss="kullback_leibler_divergence")

  log.info("Training model")
  encoding_trainer.fit(
      perturbed,
      originals,
      batch_size=128,
      epochs=20,
      verbose=0 if disable_pbar else 1)

  log.info("Extracting embeddings")
  embedding = Model(inputs=[perturbed_input_layer], outputs=[encoding_layer])
  indices = [idx for idx in idx2sample]
  samples = np.array([idx2sample[idx] for idx in indices])
  embeddings = embedding.predict(samples)
  return [(idx, embeddings[row, :]) for row, idx in enumerate(indices)]


def EmbedAutoEncoder(
    hypergraph,
    dimension,
    run_in_parallel=True,
    disable_pbar=False):
  workers = multiprocessing.cpu_count() if run_in_parallel else 1

  def do_half(important_rows, matrix):
    samples = []
    with Pool(workers,
              initializer=_init_auto_encoder_sample,
              initargs=(matrix,
                       )) as pool:
      for tmp in tqdm(pool.imap(_auto_encoder_sample,
                                important_rows),
                      total=len(important_rows),
                      disable=disable_pbar):
        samples += tmp

    return _get_auto_encoder_embeddings(
        samples,
        dimension,
        _to_samples(important_rows,
                    matrix),
        disable_pbar)

  embedding = HypergraphEmbedding()
  embedding.dim = dimension
  embedding.method_name = "AutoEncoder"

  log.info("Collecting node samples for auto encoder")
  for node_idx, emb in do_half(hypergraph.node, ToCsrMatrix(hypergraph)):
    embedding.node[node_idx].values.extend(emb)

  log.info("Collecting edge samples for auto encoder")
  for edge_idx, emb in do_half(hypergraph.edge, ToEdgeCsrMatrix(hypergraph)):
    embedding.edge[edge_idx].values.extend(emb)

  return embedding
