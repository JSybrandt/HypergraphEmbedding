#!/usr/bin/env python3

"""
Okay, the previous code at least produces SOMETHING.
So this code is going to create a hypergraph result that parses out the result
file, in order to create a new result proto. This will let us run the
experiments using the original runner.
"""

import argparse
from pathlib import Path
import logging
import logging.handlers
from hypergraph_embedding import ExperimentalResult
from tqdm import tqdm

log = logging.getLogger()

def ParseArgs():
  parser = argparse.ArgumentParser(
      description="Converts hypergraphs to metapath2vec")
  parser.add_argument(
      "-r",
      "--result",
      type=str,
      help="Path to the original result proto used in the conversion")
  parser.add_argument(
      "-t",
      "--vector-text",
      type=str,
      help="Path to vector text file (output of metapath2vec)")
  parser.add_argument(
      "-o",
      "--out",
      type=str,
      help="Path to result proto")

  parser.add_argument(
      "--log-level",
      type=str,
      help=("Specifies level of logging verbosity. " "Options: CRITICAL, ERROR, WARNING, INFO, DEBUG, NONE"),
      default="INFO")
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
  handler = logging.StreamHandler()
  handler.setFormatter(formatter)
  log.addHandler(handler)


if __name__ == "__main__":
  args = ParseArgs()
  ConfigureLogger(args)

  log.info("Checking result path")
  original_result_path = Path(args.result)
  assert original_result_path.is_file()

  log.info("Checking vector text")
  vector_text_path = Path(args.vector_text)
  assert vector_text_path.is_file()

  log.info("Checking output location is writeable")
  output_path = Path(args.out)
  assert not output_path.exists()
  assert output_path.parent.is_dir()

  log.info("Parsing input result")
  try:
    result = ExperimentalResult()
    with original_result_path.open('rb') as proto:
      result.ParseFromString(proto.read())
  except:
    log.error("Invalid input proto %s", original_result_path)

  log.info("Clearing original unimportant data")
  result.embedding.Clear()
  result.embedding.method_name = "metapath2vec++"

  log.info("Parsing vector")
  with vector_text_path.open('r') as input_file:
    for line in tqdm(input_file):
      if line[0] in 'va':
        is_edge = line[0] == 'v'
        tokens = line.strip().split()
        idx = int(tokens[0][1:])
        vector = [float(t) for t in tokens[1:]]
        if is_edge:
          half_emb = result.embedding.edge
        else:
          half_emb = result.embedding.node
        half_emb[idx].values.extend(vector)
        if result.embedding.HasField('dim'):
          assert result.embedding.dim == len(vector)
        else:
          result.embedding.dim = len(vector)


  log.info("Checking result")
  for node_idx in result.hypergraph.node:
    assert node_idx in result.embedding.node
  for edge_idx in result.hypergraph.edge:
    assert edge_idx in result.embedding.edge

  log.info("Writing result")
  with output_path.open('wb') as output_file:
    output_file.write(result.SerializeToString())

