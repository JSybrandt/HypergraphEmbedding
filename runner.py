#!/usr/bin/env python3

# This file runs the hypergraph embedding and experiments

import argparse
import logging
import logging.handlers
from pathlib import Path
from hypergraph_embedding import *

log = logging.getLogger()

global EMBEDDING_OPTIONS
global PARSING_OPTIONS
global EXPERIMENT_OPTIONS


def ParseArgs():
  parser = argparse.ArgumentParser(
      description="Process hypergraph and run experiments")

  # Logging
  parser.add_argument(
      "--log-level",
      type=str,
      help=("Specifies level of logging verbosity. "
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
      help=("Specifies how to parse the input file. "
            "Options: " + " ".join([o for o in PARSING_OPTIONS])),
      default="SNAP")
  parser.add_argument(
      "--name",
      type=str,
      help="Name of resulting hypergraph. If none, uses path",
      default="")

  # Embedding options (convert hypergraph to embedding)
  parser.add_argument(
      "--embedding",
      type=str,
      help=("Path to store embedding. "
            "Result is an embedding proto."))
  parser.add_argument(
      "--embedding-method",
      type=str,
      help=("Specifies the manner in which the provided hypergraph should be "
            "embedded. Options: " + " ".join([o for o in EMBEDDING_OPTIONS])))
  parser.add_argument(
      "--embedding-dimension",
      type=int,
      help=("Dimensonality of output embeddings. "
            "Should be positive and less than #nodes and #edges."))
  parser.add_argument(
      "--embedding-debug-summary",
      type=str,
      help=(
          "If set, in combination with an appropriate embedding-method we will "
          "write a summary of our embedding information. For instance, we may "
          "provided a histogram of each sampled probabilities"))

  # experiment options
  parser.add_argument(
      "--experiment",
      type=str,
      help=("If set, perform an evaluation experiment on the hypergraph. "
            "--experiment-result must also be set. "
            "Options: " + " ".join([o for o in EXPERIMENT_OPTIONS])),
      nargs="*")
  parser.add_argument(
      "--experiment-result",
      type=str,
      help=("Path to store experiment data proto. If set --experiment "
            "must also be set."))
  parser.add_argument(
      "--experiment-lp-probability",
      type=float,
      help=("Used to determine the proportion of removed node-edge connections "
            "for LP_* experiments"),
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

  # check for writing embedding
  if args.embedding:
    log.info("Performing checks for writing embedding")
    embedding_path = Path(args.embedding)
    log.info("Checking for valid embedding-method")
    assert args.embedding_method in EMBEDDING_OPTIONS
    log.info("Checking for positive dimension")
    assert args.embedding_dimension > 0
    log.info("Checking its safe to write embedding")
    assert not embedding_path.exists()
    assert embedding_path.parent.is_dir()

  log.info("Checking that experimental flags are appropriate")
  log.info("Checking that if experiment is set, experiment-result is too")
  assert bool(args.experiment) == bool(args.experiment_result)
  if args.experiment is not None:
    log.info("Checking that embedding is also specified")
    assert args.embedding_method
    log.info("Checking for --experiment-result")
    experiment_result_path = Path(args.experiment_result)
    assert not experiment_result_path.exists()
    assert experiment_result_path.parent.is_dir()
    log.info("Checking that experiment is valid")
    for experiment in args.experiment:
      assert experiment in EXPERIMENT_OPTIONS

  if args.embedding_debug_summary:
    log.info("--embedding_debug_summary set, checking for appropriate method")
    assert args.embedding_method in DEBUG_SUMMARY_OPTIONS
    log.info("Ensuring that its safe to write debug summary")
    embedding_debug_summary_path = Path(args.embedding_debug_summary)
    assert not embedding_debug_summary_path.exists()
    assert embedding_debug_summary_path.parent.is_dir()

  log.info("Finished checking, lgtm")

  # do conversion
  if args.raw_data:
    hypergraph = ParseRawIntoHypergraph(args, raw_data_path)
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

  log.info("Hypergraph contains %i nodes and %i edges", len(hypergraph.node),
           len(hypergraph.edge))

  if args.embedding:
    log.info("Writing an embedding for FULL input hypergraph")
    embedding = Embed(args, hypergraph)
    with embedding_path.open('wb') as proto:
      proto.write(embedding.SerializeToString())

  if args.experiment:
    lp_data = PrepLinkPredictionExperiment(hypergraph, args)
    result = LinkPredictionDataToResultProto(lp_data)
    for experiment in args.experiment:
      log.info("Performing %s experiment", experiment)
      metric = RunLinkPredictionExperiment(lp_data, experiment)
      new_metric = result.metrics.add()
      new_metric.ParseFromString(metric.SerializeToString())
    with experiment_result_path.open('wb') as proto:
      proto.write(result.SerializeToString())

  log.info("Done!")
