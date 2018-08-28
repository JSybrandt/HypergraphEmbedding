#!/usr/bin/env python3

# This file runs the hypergraph embedding and experiments

import argparse
import logging
import logging.handlers
from pathlib import Path
from hypergraph_embedding import *

log = logging.getLogger()

EMBEDDING_OPTIONS = {"SVD": EmbedSvd, "Random": EmbedRandom}


def ParseArgs():
  parser = argparse.ArgumentParser(
      description="Process hypergraph and run experiments")
  parser.add_argument(
      "--aminer-data",
      type=str,
      help="Text file from AMiner Academic Social Network.",
      default="")
  parser.add_argument(
      "--log-level",
      type=str,
      help=(
          "Specifies level of logging verbosity."
          " Options: CRITICAL, ERROR, WARNING, INFO, DEBUG, NONE"),
      default="INFO")
  parser.add_argument(
      "--log-to-stderr",
      action="store_true",
      help="If set, also print logs to stdout")
  parser.add_argument(
      "--embedding-method",
      type=str,
      help=(
          "Specifies the manner in which the provided hypergraph should be "
          "embedded. Options: " + " ".join([o for o in EMBEDDING_OPTIONS])),
      default="SVD")
  parser.add_argument(
      "hypergraph",
      type=str,
      help=(
          "Path to stored hypergraph proto. "
          "Used as output file if aminer-data is supplied."))
  # parser.add_argument(
  # "embedding",
  # type=str,
  # help=(
  # "Path to store resulting embedding."
  # "Result is an embedding proto."))
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

  formatter = logging.Formatter(
      "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

  if args.log_to_stderr:
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    log.addHandler(handler)


def ConvertAMinerToHypergraph(input_path, output_path):
  "Reads text from input_path and writes serialized proto to output_path"
  with input_path.open('r') as aminer, output_path.open('wb') as proto:
    hypergraph = PapersToHypergraph(ParseAMiner(aminer))
    log.info("Writing Hypergraph")
    proto.write(hypergraph.SerializeToString())


def LoadHypergraph(input_path):
  "Loads serialized proto from input_path"
  with input_path.open('rb') as proto:
    result = Hypergraph()
    result.ParseFromString(proto.read())
    return result


if __name__ == "__main__":
  args = ParseArgs()
  ConfigureLogger(args)

  log.info("Starting")

  assert args.hypergraph  # must be non-empty and non-None
  hypergraph_path = Path(args.hypergraph)

  log.info("Checking that provided embedding option is supported.")
  assert args.embedding_method in EMBEDDING_OPTIONS

  if args.aminer_data:
    log.info("Reading aminer co-authorship data from %s", args.aminer_data)
    aminer_data_path = Path(args.aminer_data)
    # must have supplied a real path
    assert aminer_data_path.is_file()
    # must have NOT given me a real hypergraph path
    # Note, run again without --aminer-data to use the previously converted hypergraph
    log.info("Checking that %s does not already exist", hypergraph_path)
    assert not hypergraph_path.exists()
    log.info(
        "Writing AMiner data from %s to %s",
        aminer_data_path,
        hypergraph_path)
    ConvertAMinerToHypergraph(aminer_data_path, hypergraph_path)

  log.info("Checking that %s exists", hypergraph_path)
  assert hypergraph_path.exists()

  log.info("Reading data from %s", hypergraph_path)
  hypergraph = LoadHypergraph(hypergraph_path)

  log.info("Done!")
