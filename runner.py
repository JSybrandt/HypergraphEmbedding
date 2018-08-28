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

  # Embedding options (convert hypergraph to embedding)
  parser.add_argument(
      "--embedding",
      type=str,
      help=(
          "Path to store resulting embedding."
          "Result is an embedding proto."))
  parser.add_argument(
      "--embedding-method",
      type=str,
      help=(
          "Specifies the manner in which the provided hypergraph should be "
          "embedded. Options: " + " ".join([o for o in EMBEDDING_OPTIONS])))
  parser.add_argument(
      "--dimension",
      type=int,
      help=(
          "Dimensonality of output embeddings. "
          "Should be positive and less than #nodes and #edges."))

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
    log.setLevel(logging.NONE)

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
    assert raw_data_path.exists()
    log.info("Checking its safe to write hypergraph")
    assert not hypergraph_path.exists()
    assert hypergraph_path.parent.is_dir()

  if args.embedding:
    embedding_path = Path(args.embedding)
    log.info("Checking for valid embedding-method")
    assert args.embedding_method in EMBEDDING_OPTIONS
    log.info("Checking for positive dimension")
    assert args.dimension > 0
    if not args.raw_data:  # if we are converting also, we don't check hg here
      log.info("Checking hypergraph exists")
      assert hypergraph_path.exists()
    log.info("Checking its safe to write embedding")
    assert not embedding_path.exists()
    assert embedding_path.parent.is_dir()

  log.info("Finished checking, lgtm")

  # do conversion
  if args.raw_data:
    log.info("Parsing %s with %s method", raw_data_path, args.raw_data_format)
    with raw_data_path.open('r') as raw_file:
      hypergraph = PARSING_OPTIONS[args.raw_data_format](raw_data)
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

  if args.embedding:
    log.info("Checking embedding dimensionality is smaller than # nodes/edges")
    assert min(len(hypergraph.node), len(hypergraph.edge)) > args.dimension

    log.info("Embedding using method %s", args.embedding_method)
    embedding = EMBEDDING_OPTIONS[args.embedding_method](
        hypergraph,
        args.dimension)

    log.info("Writing embedding")
    with embedding_path.open('wb') as proto:
      proto.write(embedding.SerializeToString())

  log.info("Done!")
