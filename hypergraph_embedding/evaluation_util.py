# This file provides evaluation utilities for hypergraph experiments

from . import Hypergraph, EvaluationMetrics, ExperimentalResult
from .hypergraph_util import *
from .embedding import Embed
from collections import namedtuple
from itertools import product
import logging
import numpy as np
import multiprocessing
from multiprocessing import Pool, Manager
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from random import random, choice, sample
import warnings

log = logging.getLogger()

global EXPERIMENT_OPTIONS

# Constant used to share information across processors.
# Each map will call an _init* function that sets values in
# this dict.
_k_shared_data = {}


def LinkPredictionExperiment(args, hypergraph, predictor):
  """
  This function sets up a link-prediction experiment."
  First we sample the given hypergraph, then we embed it using details from"
  args. Finally we use predictor to evaluate samples and get metrics."
  inputs:
    - args: parsed arguments from main
    - hypergraph: a hypergraph proto message to be evaluated
    - predictor: a function
      - inputs:
        - hypergraph: Sampled hypergraph proto message
        - embedding: embedding proto corresponding to sampled hypergraph
        - links: list of pos+neg node-edge samples
      - outputs:
        - predicted_links: list of links filtered from input links
                           only outputs node-edge links that are "true"
  """

  log.info("Checking that --experiment-lp-probabilty is between 0 and 1")
  assert args.experiment_lp_probability >= 0
  assert args.experiment_lp_probability <= 1

  log.info(
      "Creating subgraph with removal prob. %f",
      args.experiment_lp_probability)
  new_graph, good_links = RemoveRandomConnections(hypergraph, args.experiment_lp_probability)
  log.info("Removed %i links", len(good_links))

  log.info("Sampling missing links for evaluation")
  bad_links = SampleMissingConnections(hypergraph, len(good_links))
  log.info("Sampled %i links", len(bad_links))

  log.info("Embedding new hypergraph")
  embedding = Embed(args, new_graph)

  log.info("Predicting links on subset graph")
  predicted_links = predictor(new_graph, embedding, bad_links + good_links)

  log.info("Evaluting link prediction performance")
  metrics = CalculateCommunityPredictionMetrics(
      predicted_links,
      good_links,
      bad_links)

  log.info("Result:\n%s", metrics)

  log.info("Storing data into Experimental Result proto")
  res = ExperimentalResult()
  res.metrics.ParseFromString(metrics.SerializeToString())
  res.hypergraph.ParseFromString(new_graph.SerializeToString())
  res.embedding.ParseFromString(embedding.SerializeToString())
  res.experiment_name = args.experiment_type
  return res


def RemoveRandomConnections(original_hypergraph, probability):
  """
    This function creates a new hypergraph where each node-edge with randomly
    sampled nodes removed from communities.
    input:
      - hypergraph proto object
      - number between 0 and 1
    outut:
      - new hypergraph with removed edges
      - list of all removed node-edge pairs
    """
  assert probability >= 0
  assert probability <= 1

  # stores (node_idx, edge_idx) tuples
  removed_connections = []

  new_hg = Hypergraph()
  if original_hypergraph.HasField("name"):
    new_hg.name = original_hypergraph.name
  for node_idx, node in original_hypergraph.node.items():
    for edge_idx in node.edges:
      edge = original_hypergraph.edge[edge_idx]

      # If we are going to drop this connection
      if random() < probability and probability > 0:
        removed_connections.append((node_idx, edge_idx))
      else:
        AddNodeToEdge(
            new_hg,
            node_idx,
            edge_idx,
            node.name if node.HasField("name") else None,
            edge.name if edge.HasField("name") else None)
  return new_hg, removed_connections


def SampleMissingConnections(hypergraph, num_samples):
  """
    Given a hypergraph, Sample for random not-present node-edge connections.
    Used as negative training data for the link prediction task.
    input:
      - hypergraph: hypergraph proto
      - num_samples
    output:
      - list of node_idx, edge_idx pairs where node is not in edge
  """
  samples = set()
  num_nodes = len(hypergraph.node)
  num_edges = len(hypergraph.edge)
  assert num_samples < num_nodes * num_edges
  assert len(hypergraph.edge) > 0
  assert len(hypergraph.node) > 0

  nodes = [n for n in hypergraph.node]
  edges = [e for e in hypergraph.edge]

  num_tries_till_failure = 10 * num_samples
  while len(samples) < num_samples and num_tries_till_failure:
    num_tries_till_failure -= 1
    node_idx = choice(nodes)
    edge_idx = choice(edges)
    if edge_idx not in hypergraph.node[node_idx].edges:
      samples.add((node_idx, edge_idx))

  if len(samples) < num_samples:
    log.critical(
        "SampleMissingConnections failed to find %i samples",
        num_samples)
  return list(samples)


def CalculateCommunityPredictionMetrics(
    predicted_connections,
    good_links,
    bad_links):
  """
    Treats good_links as positive examples, and computes a Metrics
    named tuple.
    input:
      - predicted_connections: iterable of (node_idx, edge_idx) pairs
      - good_links: iterable of (node_idx, edge_idx) pairs from original
      - bad_links: iterable of (node_idx, edge_idx) not from original
    output:
      - EvaluationMetrics proto containing all fields but accuracy
        and num_true_neg
    """
  predictions = set(predicted_connections)
  positives = set(good_links)
  negatives = set(bad_links)

  # + and - must be disjoin
  assert len(positives.intersection(negatives)) == 0

  # predictions must be a subset of the + and - samples
  assert len(predictions.intersection(
      positives.union(negatives))) == len(predictions)

  assert len(positives) + len(negatives) > 0

  true_positives = predictions.intersection(positives)

  metrics = EvaluationMetrics()

  if len(predictions):
    metrics.precision = len(true_positives) / len(predictions)

  if len(positives):
    metrics.recall = len(true_positives) / len(positives)

  if metrics.precision + metrics.recall:
    metrics.f1 = 2 * metrics.precision * metrics.recall / (
        metrics.precision + metrics.recall)

  metrics.num_true_pos = len(true_positives)
  metrics.num_false_pos = len(predictions) - len(true_positives)
  metrics.num_false_neg = len(positives) - len(true_positives)
  metrics.num_true_neg = len(negatives - predictions)
  metrics.accuracy = (metrics.num_true_pos + metrics.num_true_neg) / (
      len(positives) + len(negatives))
  return metrics


################################################################################
# EdgeCentroidPrediction - Helper Functions                                    #
################################################################################


def _init_node_emb_to_numpy(_embedding):
  _k_shared_data['embedding'] = _embedding


def _node_emb_to_numpy(node_idx):
  return (
      node_idx,
      np.asarray(
          _k_shared_data['embedding'].node[node_idx].values,
          dtype=np.float32))


def _init_get_edge_centroid_range(
    _node2embedding,
    _hypergraph,
    _distance_function):
  _k_shared_data['node2embedding'] = _node2embedding
  _k_shared_data['hypergraph'] = _hypergraph
  _k_shared_data['distance_function'] = _distance_function


def _get_edge_centroid_range(edge_idx):
  points = [
      _k_shared_data['node2embedding'][i]
      for i in _k_shared_data['hypergraph'].edge[edge_idx].nodes
  ]
  centroid = np.mean(points, axis=0)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    max_dist = max(
        [_k_shared_data['distance_function'](centroid,
                                             vec) for vec in points])
  return (edge_idx, centroid, max_dist)


def _init_is_node_in_sphere(
    _node2embedding,
    _edge2centroid,
    _edge2range,
    _hypergraph,
    _distance_function):
  _k_shared_data['node2embedding'] = _node2embedding
  _k_shared_data['edge2centroid'] = _edge2centroid
  _k_shared_data['edge2range'] = _edge2range
  _k_shared_data['hypergraph'] = _hypergraph
  _k_shared_data['distance_function'] = _distance_function


def _is_node_in_sphere(indices):
  "Checks if this node is in any of the edges"
  node_idx, edge_idx = indices
  if edge_idx in _k_shared_data['hypergraph'].node[node_idx].edges:
    return None
  if node_idx not in _k_shared_data['node2embedding']:
    return None
  if edge_idx not in _k_shared_data['edge2centroid']:
    return None

  vec = _k_shared_data['node2embedding'][node_idx]
  centroid = _k_shared_data['edge2centroid'][edge_idx]

  # cosine distance might cause errors if we have a 0 vector
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if _k_shared_data['distance_function'](
        vec,
        centroid) <= _k_shared_data['edge2range'][edge_idx]:
      return (node_idx, edge_idx)

  return None


def EdgeCentroidPrediction(
    hypergraph,
    embedding,
    links,
    distance_function=cosine,
    run_in_parallel=True):
  """
  Given a hypergraph (assumed to contain missing links) and a corresponding
  embedding, idetify missing node-edge connections. Performs this task by
  comparing cosine similarities of potential connections with existing ones.
  Note, drops edges with < 2 nodes.
  Note, all nodes must be embedded.
  input:
    - hypergraph: Hypergraph proto
    - embedding: Hypergraph proto
    - distance_function: mesure of distance between embeddings
    - links: the set of node_idx, edge_idx pairs to evaluate
  output:
    - list of node-edge pairs from links predicted to be accurate
  """

  assert embedding.dim > 0

  num_cores = multiprocessing.cpu_count() if run_in_parallel else 1

  log.info("Converting all node embeddings to numpy")
  with Pool(num_cores,
            initializer=_init_node_emb_to_numpy,
            initargs=(embedding,
                     )) as pool:
    node2embedding = {
        x[0]: x[1] for x in pool.map(_node_emb_to_numpy,
                                     embedding.node)
    }

  log.info("Finding edges with at least two nodes")
  edges = [idx for idx, edge in hypergraph.edge.items() if len(edge.nodes) >= 2]

  log.info("Identifying each edge's centroid and range")

  with Pool(num_cores,
            initializer=_init_get_edge_centroid_range,
            initargs=(node2embedding,
                      hypergraph,
                      distance_function)) as pool:
    edge_centroid_range = pool.map(_get_edge_centroid_range, edges)

  log.info("Indexing edges")
  edge2centroid = {x[0]: x[1] for x in edge_centroid_range}
  edge2range = {x[0]: x[2] for x in edge_centroid_range}

  predicted_links = []

  log.info("Calculating whether each point is in each edge's range")
  with Pool(num_cores,
            initializer=_init_is_node_in_sphere,
            initargs=(node2embedding,
                      edge2centroid,
                      edge2range,
                      hypergraph,
                      distance_function)) as pool:
    for res in pool.imap(_is_node_in_sphere, links, chunksize=250):
      if res:
        predicted_links.append(res)

  return predicted_links


################################################################################
# EdgeClassifierPrediction - Helper functions                                  #
################################################################################

## These stub classes are used for edge cases where we don't want a typical   ##
## Classifier                                                                 ##


class LetEverythingIn():

  def predict(*args, **kargs):
    return 1


class LetNothingIn():

  def predict(*args, **kargs):
    return 0


def _init_evaluate_classifier(_idx2embedding, _idx2classifier):
  _k_shared_data['idx2embedding'] = _idx2embedding
  _k_shared_data['idx2classifer'] = _idx2classifier


def _evaluate_classifier(indices):
  "Checks if this node is in any of the edges"
  emb_idx, classifier_idx = indices
  node_vec = _k_shared_data['idx2embedding'][emb_idx]
  edge_model = _k_shared_data['idx2classifer'][classifier_idx]

  return indices, edge_model.predict([node_vec])[0]


def _init_train_personalized_classifier(
    _idx2neighbors,
    _neighbor_idx2embedding):
  _k_shared_data['idx2neighbors'] = _idx2neighbors
  _k_shared_data['neighbor_idx2embedding'] = _neighbor_idx2embedding


def _train_personalized_classifier(idx):

  # Sample positive results
  pos_indices = set(_k_shared_data["idx2neighbors"][idx])
  if (len(pos_indices) == 0):
    return (idx, LetNothingIn())

  neg_indices = _k_shared_data["neighbor_idx2embedding"].keys() - pos_indices
  if (len(neg_indices) == 0):
    return (idx, LetEverythingIn())

  # sample negative results to equal positive
  neg_indices = sample(neg_indices, min(len(neg_indices), len(pos_indices)))

  assert len(pos_indices) > 0
  assert len(neg_indices) > 0

  samples = []
  labels = []
  for neigh_idx in pos_indices:
    samples.append(_k_shared_data["neighbor_idx2embedding"][neigh_idx].values)
    labels.append(1)
  for neigh_idx in neg_indices:
    samples.append(_k_shared_data["neighbor_idx2embedding"][neigh_idx].values)
    labels.append(0)
  samples, labels = shuffle(samples, labels)
  return (idx, LinearSVC().fit(samples, labels))


def GetPersonalizedClassifiers(
    hypergraph,
    embedding,
    per_edge=True,
    idx_subset=None,
    run_in_parallel=True):
  """
  Returns a dict from idx-classifier.
  If per_edge=True then there will be |edges| classifiers each mapping
  node embedding to boolean. If per_edge=False, then there will be
  |nodes| classifiers mapping edge embedding to boolean.
  Outputs a dict from idx to classifier.
  """

  num_cores = multiprocessing.cpu_count() if run_in_parallel else 1

  if per_edge:
    idx2neighbors = {idx: edge.nodes for idx, edge in hypergraph.edge.items()}
  else:
    idx2neighbors = {idx: node.edges for idx, node in hypergraph.node.items()}
  neighbor_idx2embedding = embedding.node if per_edge else embedding.edge

  result = {}
  log.info("Training classifier per %s", "edge" if per_edge else "node")
  if idx_subset is None:
    idx_subset = idx2neighbors.keys()
  else:
    log.info("Subset provided with %i entires", len(idx_subset))

  with Pool(num_cores,
            initializer=_init_train_personalized_classifier,
            initargs=(idx2neighbors,
                      neighbor_idx2embedding)) as pool:
    with tqdm(total=len(idx_subset)) as pbar:
      for idx, classifier in pool.imap(_train_personalized_classifier, idx_subset):
        result[idx] = classifier
        pbar.update(1)
  return result


def PersonalizedClassifierPrediction(
    hypergraph,
    embedding,
    links,
    per_edge=True,
    run_in_parallel=True):
  """
  Given a hypergraph (assumed to contain missing links) and a corresponding
  embedding, idetify missing node-edge connections. Performs this task by
  training a classifier per_node / per_edge and running predictions through
  each "personalized classifier"
  input:
    - hypergraph: Hypergraph proto
    - embedding: embedding proto
    - links: the set of node_idx, edge_idx pairs to evaluate
    - per_edge: whether to train a classifer for each edge or node
  output:
    - list of node-edge pairs from links predicted to be accurate
  """

  assert embedding.dim > 0

  num_cores = multiprocessing.cpu_count() if run_in_parallel else 1

  log.info("Removing potential links that do not have embeddings")
  entity_idx_classifier_idx = [(n,
                                e) if per_edge else (e,
                                                     n)
                               for n,
                               e in links
                               if n in embedding.node and e in embedding.edge]

  # Get only the needed ones
  nessesary_classifiers = set(c for _, c in entity_idx_classifier_idx)

  log.info("Training a classifier per edge")
  idx2classifier = GetPersonalizedClassifiers(
      hypergraph,
      embedding,
      per_edge=per_edge,
      idx_subset=nessesary_classifiers,
      run_in_parallel=run_in_parallel)
  log.info("Mapping %s to embeddings", "nodes" if per_edge else "edges")
  if per_edge:
    idx2embedding = {idx: emb.values for idx, emb in embedding.node.items()}
  else:
    idx2embedding = {idx: emb.values for idx, emb in embedding.edge.items()}

  predicted_links = []

  log.info("Running each node-edge classification")
  with Pool(num_cores,
            initializer=_init_evaluate_classifier,
            initargs=(idx2embedding,
                      idx2classifier)) as pool:
    for (entity_idx, classifier_idx), res in pool.imap(_evaluate_classifier, entity_idx_classifier_idx, chunksize=250):
      if res is not None and res > 0:
        if per_edge:
          node_idx, edge_idx = (entity_idx, classifier_idx)
        else:
          edge_idx, node_idx = (entity_idx, classifier_idx)
        predicted_links.append((node_idx, edge_idx))

  return predicted_links


################################################################################
# NodeEdgeClassifierPrediction - Train a binary classifier for node/edge emb.  #
################################################################################


def _GetVectorFromIdx(node_idx, edge_idx, embedding):
  assert node_idx in embedding.node
  assert edge_idx in embedding.edge
  return np.concatenate(
      (embedding.node[node_idx].values,
       embedding.edge[edge_idx].values),
      axis=0)


def _NodeEdgeVectors(hypergraph, embedding):
  for node_idx, node in hypergraph.node.items():
    for edge_idx in node.edges:
      yield _GetVectorFromIdx(node_idx, edge_idx, embedding)


def _TrainNodeEdgeEmbeddingClassifier(hypergraph, embedding):
  """
  Returns a classifier trained to predict node-edge connections biased
  on the provided hypergraph and embedding.
  Output:
    - A model that impliments a `predict` method, mapping
      [node_embedding edge_embedding] to 0 or 1
  """

  log.info("Collecting postive training examples")
  examples = []
  labels = []
  for vec in _NodeEdgeVectors(hypergraph, embedding):
    examples.append(vec)
    labels.append(1)

  log.info("Collecting negative training examples")
  for node_idx, edge_idx in SampleMissingConnections(hypergraph, len(examples)):
    examples.append(_GetVectorFromIdx(node_idx, edge_idx, embedding))
    labels.append(0)

  log.info("Training node-edge embedding classifier")
  examples, labels = shuffle(examples, labels)
  # I've selected a MLP classifier with 1 hidden layer equal to the embedding length
  # This means, for the classifier to succeed it must discover a linear mapping
  # from each embedding space into a shared space of equal size.
  return MLPClassifier((embedding.dim,)).fit(examples, labels)


def NodeEdgeEmbeddingPrediction(
    hypergraph,
    embedding,
    potential_links,
    classifier=None):
  """
    Returns a subset of the input potential_links that a binary classifier
    deems good. If classifier is set, we will use that instead of
    training our own (intended for testing purposes).
    Inputs:
      - hypergraph: a Hypergraph proto message
      - embedding: a HypergraphEmbedding proto message
      - potential_links: an iterable of (node_idx, edge_idx) pairs
      - (optional) classifier: if set, use this.
                               Must define predict(x)
    Output:
      - An iterable of (node_idx, edge_idx) pairs.
        These correspond to the predicted links, and will be a subset of
        potential_links
    """
  if classifier is None:
    classifier = _TrainNodeEdgeEmbeddingClassifier(hypergraph, embedding)

  log.info("Deleting input edges that are not represented in the subgraph")
  potential_links = [(n,
                      e)
                     for n,
                     e in potential_links
                     if n in hypergraph.node and e in hypergraph.edge]

  log.info("Converting potential links to embedding vectors")
  input_data = []
  for node_idx, edge_idx in tqdm(potential_links):
    input_data.append(_GetVectorFromIdx(node_idx, edge_idx, embedding))

  log.info("Evaluating predictions")
  predicted_links = []
  for link, prediction in zip(potential_links, classifier.predict(input_data)):
    if prediction == 1:
      predicted_links.append(link)
  return predicted_links


################################################################################
# Details for runner - Includes wrappers for experiment types                  #
################################################################################


def _EdgeCentroidExperiment(args, hypergraph):
  return LinkPredictionExperiment(args, hypergraph, EdgeCentroidPrediction)


def _EdgeClassifiersExperiment(args, hypergraph):
  predictor = lambda h, e, p: PersonalizedClassifierPrediction(h, e, p, per_edge=True)
  return LinkPredictionExperiment(args, hypergraph, predictor)


def _NodeClassifiersExperiment(args, hypergraph):
  predictor = lambda h, e, p: PersonalizedClassifierPrediction(h, e, p, per_edge=False)
  return LinkPredictionExperiment(args, hypergraph, predictor)


def _NodeEdgeEmbeddingExperiment(args, hypergraph):
  predictor = lambda h, e, p: NodeEdgeEmbeddingPrediction(h, e, p)
  return LinkPredictionExperiment(args, hypergraph, predictor)


EXPERIMENT_OPTIONS = {
    "LP_EDGE_CENTROID": _EdgeCentroidExperiment,
    "LP_EDGE_CLASSIFIERS": _EdgeClassifiersExperiment,
    "LP_NODE_CLASSIFIERS": _NodeClassifiersExperiment,
    "LP_NODE_EDGE_CLASSIFIER": _NodeEdgeEmbeddingExperiment
}
