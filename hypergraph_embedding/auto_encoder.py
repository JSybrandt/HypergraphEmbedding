################################################################################
# AutoEncoder with helper functions                                            #
################################################################################


def _init_auto_encoder_sample(matrix):
  _shared_info.clear()
  _shared_info["matrix"] = matrix

def _auto_encoder_sample(row_idx, matrix=None):
  if matrix is None:
    matrix = _shared_info["matrix"]
  sparse_row = matrix[row_idx]
  normalized_row = (sparse_row / sparse_row.nnz).todense()
  samples = [(normalized_row, normalized_row)]
  if sparse_row.nnz > 1:
    for col_idx in sparse_row.nonzero()[1]:
      sparse_copy = sparse_row.copy()
      sparse_copy[col_idx] = 0
      samples.append(((sparse_copy/sparse_copy.nnz).todense(),
                      normalized_row))
  return samples


def _to_samples(indices, matrix):
  return {idx, (matrix[idx]/matrix[idx].nnz).todense()
          for idx in indices}

def _get_auto_encoder_embeddings(samples, dimension, idx2sample):
  log.info("Constructing model")
  sample_size = samples[0][0].shape[1]
  permuted_sample = Input((sample_size), name="permuted_sample", dtype=np.int32)
  encoding = Dense(dimension, name="encoding", activation="softmax")(permuted_sample)
  original_sample = Dense(sample_size, name="original_sample", activation="softmax")(encoding)
  encoding_trainer = Model(inputs=[permuted_sample], outputs=[original_sample])
  encoding_trainer.compile(optimizer="adagrad", loss="kullback_leibler_divergence")

  log.info("Training model")
  input_samples = [s for s, _, in samples]
  output_samples = [s for _, s, in samples]
  encoding_trainer.fit(input_samples,
                       output_samples,
                       batch_size=128,
                       epochs=20,
                       verbose=0 if disable_pbar else 1)

  log.info("Extracting embeddings")
  embedding = Model(inputs=[permuted_sample], outputs=[encoding])
  embedding.compile()
  return {idx, embedding.predict(sample) for idx, sample in idx2sample}

def EmbedAutoEncoder(hypergraph,
                     dimension,
                     run_in_parallel=True,
                     disable_pbar=False):
  workers = multiprocessing.cpu_count() if run_in_parallel else 1

  def do_half(important_rows, matrix):
    samples = []
    with Pool(workers,
              initializer=_init_auto_encoder_sample,
              initargs=(matrix,)) as pool:
      for tmp in tqdm(pool.imap(_auto_encoder_sample, important_rows),
                          total=len(important_rows),
                          disable=disable_pbar):
        samples += tmp

    return _get_auto_encoder_embeddings(
        samples,
        dimension,
        _to_samples(important_rows, matrix))

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
