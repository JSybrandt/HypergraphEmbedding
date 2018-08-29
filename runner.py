#!/usr/bin/env python3

# This file runs the hypergraph embedding and experiments

import argparse
import logging
import logging.handlers
from pathlib import Path
from hypergraph_embedding import *

log = logging.getLogger()

EMBEDDING_OPTIONS = {"SVD": EmbedSvd, "RANDOM": EmbedRandom}

PARSING_OPTIONS = {
    "AMINER": AMinerToHypergraph,
    "SNAP": SnapCommunityToHypergraph
}

EXPERIMENT_OPTIONS = {
    "LINK_PREDICTION": RunLinkPrediction,
    "LINK_RECALL": RunLinkPrediction
}


def ParseArgs():
  parser = argparse.ArgumentParser(
      description="Process hypergraph and run experiments")

  # Logging
  parser.add_argument(
      "--log-level",
      type=str,
      help=(
          "Specifies level of logging verbosity. "
          "Options: CRITICAL, ERROR, WARNING, INFO, DEBUG, NONE"),
      default="INFO")

  # Raw data options (convert to hypergraph)
  parser.add_argument(
      "--raw-data",
      type=str,
      help="Raw data to be converted to a hypergraph before processing.",
      default="")
  parser.add_argument(
      "--raw-data-format",
      type=str,
      help=(
          "Specifies how to parse the input file. "
          "Options: " + " ".join([o for o in PARSING_OPTIONS])),
      default="")
  parser.add_argument(
      "--name",
      type=str,
      help="Name of resulting hypergraph. If none, uses path",
      default="")

  # Embedding options (convert hypergraph to embedding)
  parser.add_argument(
      "--embedding",
      type=str,
      help=(
          "Path to store / read embedding. "
          "Result is an embedding proto. "
          "Writes if --embedding-method is also specified."))
  parser.add_argument(
      "--embedding-method",
      type=str,
      help=(
          "Specifies the manner in which the provided hypergraph should be "
          "embedded. Options: " + " ".join([o for o in EMBEDDING_OPTIONS])))
  parser.add_argument(
      "--embedding-dimension",
      type=int,
      help=(
          "Dimensonality of output embeddings. "
          "Should be positive and less than #nodes and #edges."))

  # experiment options
  parser.add_argument(
      "--experiment-type",
      type=str,
      help=(
          "If set, perform an evaluation experiment on the hypergraph. "
          "--experiment-result must also be set. "
          "Options: " + " ".join([o for o in EXPERIMENT_OPTIONS])))
  parser.add_argument(
      "--experiment-result",
      type=str,
      help=(
          "Path to store experiment result proto. If set --experiment-type "
          "must also be set."))
  parser.add_argument(
      "--experiment-lp-probability",
      type=float,
      help=(
          "If --experiment-type=LINK_PREDICTION then this flag specifies "
          "the probabilty of removing a node from an edge."),
      default=0.1)

  # Required hypergraph argument
  parser.add_argument(
      "hypergraph",
      type=str,
      help=(
          "Path to stored hypergraph proto. "
          "If raw-data is supplied, this program will write a hypergraph "
          "proto here. If embedding is specified, this program will read from "
          "here. Both may be specified for full pipeline."))
  return parser.parse_args()


def ConfigureLogger(args):
  if args.log_level == "CRITICAL":
    log.setLevel(logging.CRITICAL)
  elif args.log_level == "ERROR":
    log.setLevel(logging.ERROR)
  elif args.log_level == "WARNING":
    log.setLevel(logging.WARNING)
  elif args.log_level == "INFO":
    log.setLevel(logging.INFO)
  elif args.log_level == "DEBUG":
    log.setLevel(logging.DEBUG)
  elif args.log_level == "NONE":
    log.propogate = False

  formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")

  # log to stderr
  handler = logging.StreamHandler()
  handler.setFormatter(formatter)
  log.addHandler(handler)


if __name__ == "__main__":
  args = ParseArgs()
  ConfigureLogger(args)

  log.info("Validating input arguments")
  hypergraph_path = Path(args.hypergraph)

  if args.raw_data:
    log.info("Checking for valid raw-data-format")
    assert args.raw_data_format in PARSING_OPTIONS
    raw_data_path = Path(args.raw_data)
    log.info("Checking raw-data exists")
    assert raw_data_path.is_file()
    log.info("Checking its safe to write hypergraph")
    assert not hypergraph_path.exists()
    assert hypergraph_path.parent.is_dir()
  else:
    log.info("Checking hypergraph exists")
    assert hypergraph_path.is_file()

  # check for writing
  if args.embedding and args.embedding_method:
    log.info("Performing checks for writing embedding")
    embedding_path = Path(args.embedding)
    log.info("Checking for valid embedding-method")
    assert args.embedding_method in EMBEDDING_OPTIONS
    log.info("Checking for positive dimension")
    assert args.embedding_dimension > 0
    log.info("Checking its safe to write embedding")
    assert not embedding_path.exists()
    assert embedding_path.parent.is_dir()

  elif args.embedding:  # only embedding
    log.info("Performing checks for reading embedding")
    embedding_path = Path(args.embedding)
    assert embedding_path.is_file()
    if args.embedding_dimension:
      log.warning("Dimension set, but we are not writing hypergraph")

  log.info("Checking that experimental flags are appropriate")
  log.info("Checking that if experiment-type is set, experiment-result is too")
  assert bool(args.experiment_type) == bool(args.experiment_result)
  if args.experiment_type:
    log.info("Checking that embedding is also specified")
    assert args.embedding
    log.info("Checking that --experiment-result is safe to write")
    experiment_result_path = Path(args.experiment_result)
    assert not experiment_result_path.exists()
    assert experiment_result_path.parent.is_dir()
    log.info("Checking that experiment-type is valid")
    assert args.experiment_type in EXPERIMENT_OPTIONS
    if args.experiment_type == "LINK_PREDICTION":
      log.info("Checking that --experiment-lp-probabilty is between 0 and 1")
      assert args.experiment_lp_probability >= 0
      assert args.experiment_lp_probability <= 1

  log.info("Finished checking, lgtm")

  # do conversion
  if args.raw_data:
    log.info("Parsing %s with %s method", raw_data_path, args.raw_data_format)
    with raw_data_path.open('r') as raw_file:
      hypergraph = PARSING_OPTIONS[args.raw_data_format](raw_file)
    if args.name:
      log.info("Setting hypergraph name to %s", args.name)
      log.info("Good name!")
      hypergraph.name = args.name
    else:
      log.info("Setting hypergraph name to %s", args.hypergraph)
      log.info("Bad name :(")
      hypergraph.name = args.hypergraph
    log.info("Writing hypergraph proto to %s", hypergraph_path)
    with hypergraph_path.open('wb') as proto_file:
      proto_file.write(hypergraph.SerializeToString())
    log.info("Checking write went well")
    assert hypergraph_path.exists()
  else:
    log.info("Reading hypergraph from %s", hypergraph_path)
    with hypergraph_path.open('rb') as proto_file:
      hypergraph = Hypergraph()
      hypergraph.ParseFromString(proto_file.read())

  log.info(
      "Hypergraph contains %i nodes and %i edges",
      len(hypergraph.node),
      len(hypergraph.edge))

  if args.embedding and args.embedding_method:
    log.info("Checking embedding dimensionality is smaller than # nodes/edges")
    assert min(len(hypergraph.node),
               len(hypergraph.edge)) > args.embedding_dimension

    log.info("Embedding using method %s", args.embedding_method)
    embedding = EMBEDDING_OPTIONS[args.embedding_method](
        hypergraph,
        args.embedding_dimension)

    log.info("Writing embedding")
    with embedding_path.open('wb') as proto:
      proto.write(embedding.SerializeToString())
  elif args.embedding:
    embedding = HypergraphEmbedding()
    log.info("Reading embedding from %s", embedding_path)
    with embedding_path.open('rb') as proto:
      embedding.ParseFromString(proto.read())

  log.info(
      "Embedding contains %i node and %i edge vectors",
      len(embedding.node),
      len(embedding.edge))
  log.info(
      "Embedding is of dimensionality %i and was built with method (%s)",
      embedding.dim,
      embedding.method_name)

  if args.experiment_type:
    log.info("Performing %s experiment", args.experiment_type)
    experiment = EXPERIMENT_OPTIONS[args.experiment_type]
    if args.experiment_type == "LINK_PREDICTION":
      metrics = experiment(
          hypergraph,
          embedding,
          args.experiment_lp_probability,
          False)
    elif args.experiment_type == "LINK_RECALL":
      metrics = experiment(
          hypergraph,
          embedding,
          args.experiment_lp_probability,
          True)
    log.info("Experiment results:")
    log.info(metrics)
    log.info("Writing metrics proto")
    with experiment_result_path.open('wb') as proto:
      proto.write(metrics.SerializeToString())

  log.info("Done!")
