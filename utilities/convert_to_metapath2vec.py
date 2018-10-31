#!/usr/bin/env python3

"""
The metapath2vec format is atrocious.
This utility is going to convert a (beautiful) hypergraph proto object
into a directory containing five files:
  - id_author.txt
  - id_conf.txt
  - paper_author.txt
  - paper_conf.txt
  - paper.txt

These should act as input to py2genMetaPaths.py
and that should create input for metapath2vec...

who knows?
"""

import argparse
from pathlib import Path
import logging
import logging.handlers
from hypergraph_embedding import ExperimentalResult

log = logging.getLogger()

def ParseArgs():
  parser = argparse.ArgumentParser(
      description="Converts hypergraphs to metapath2vec")
  parser.add_argument(
      "-r",
      "--result",
      type=str,
      help="Path to hypergraph result proto")
  parser.add_argument(
      "-o",
      "--out-dir",
      type=str,
      help="Path to result directory")
  parser.add_argument(
      "--log-level",
      type=str,
      help=("Specifies level of logging verbosity. "
            "Options: CRITICAL, ERROR, WARNING, INFO, DEBUG, NONE"),
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

def touch_and_check(path):
  if path.exists():
    log.error("Attempted to create %s, but its already there.", path)
  try:
    path.touch()
  except:
    log.error("Touch %s failed.", path)
  if not path.is_file():
    log.error("Failed to create %s.", path)
    raise RuntimeError("Failed to create " + str(path))


if __name__ == "__main__":
  args = ParseArgs()
  ConfigureLogger(args)

  log.info("Checking input path")
  result_proto_path = Path(args.result)
  assert result_proto_path.is_file()

  out_dir_path = Path(args.out_dir)
  log.info("Checking that output path does NOT exists")
  assert not out_dir_path.exists()
  log.info("Checking that output path is writable")
  assert out_dir_path.parent.is_dir()


  log.info("Attempting to read hypergraph")
  result = ExperimentalResult()
  try:
    with result_proto_path.open('rb') as proto:
      result.ParseFromString(proto.read())
  except:
    log.error("Invalid proto file: %s", result_proto_path)
    exit(1)
  hypergraph = result.hypergraph

  log.info("Creating dir and files")
  try:
    out_dir_path.mkdir()
  except:
    log.error("Failed to make %s", out_dir_path)
    exit(1)

  largest_node_idx = max(hypergraph.node)

  # Because mp2v is written with only the 3-type case in mind
  # we have to represent each node twice...
  def to_author_idx(node_idx):
    return node_idx
  def to_conf_idx(edge_idx):
    return edge_idx + largest_node_idx + 1

  id_auth_path = out_dir_path.joinpath("id_author.txt")
  id_conf_path = out_dir_path.joinpath("id_conf.txt")
  paper_auth_path = out_dir_path.joinpath("paper_author.txt")
  paper_conf_path = out_dir_path.joinpath("paper_conf.txt")
  paper_path = out_dir_path.joinpath("paper.txt")

  touch_and_check(id_auth_path)
  touch_and_check(id_conf_path)
  touch_and_check(paper_auth_path)
  touch_and_check(paper_conf_path)
  touch_and_check(paper_path)

  # it appears that everything needs a unique id
  # conference id < author id < paper id

  # MP2V is built to sample ONLY conference-author relations
  # and it ONLY embeds conferences and authors.
  # This means I need to makeup papers that link nodes and edges.
  # This is hella dumb

  log.info("Writing authors")
  with id_auth_path.open('w') as auth_file:
    for node_idx in hypergraph.node:
      auth_idx = to_author_idx(node_idx)
      auth_file.write("{}\ta{}\n".format(auth_idx, node_idx))

  log.info("Writing conferences")
  with id_conf_path.open('w') as conf_file:
    for edge_idx in hypergraph.edge:
      conf_idx = to_conf_idx(edge_idx)
      conf_file.write("{}\tv{}\n".format(conf_idx, edge_idx))

  log.info("Writing papers to link authors and conferences")
  paper_idx = largest_node_idx + max(hypergraph.edge) + 2
  with paper_path.open('w') as paper_file, \
      paper_auth_path.open('w') as paper_auth_file, \
      paper_conf_path.open('w') as paper_conf_file:
    for node_idx, node in hypergraph.node.items():
      for edge_idx in node.edges:
        paper_idx += 1  # get new paper_idx
        auth_idx = to_author_idx(node_idx)
        conf_idx = to_conf_idx(edge_idx)
        paper_auth_file.write("{}\t{}\n".format(paper_idx, auth_idx))
        paper_conf_file.write("{}\t{}\n".format(paper_idx, conf_idx))
        paper_file.write("{}\tLinks {} to {}\n".format(paper_idx,
                                                     node_idx,
                                                     edge_idx))

