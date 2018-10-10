#!/usr/bin/env python3

# This file runs the hypergraph embedding and experiments

import argparse
import logging
import logging.handlers
from pathlib import Path
from hypergraph_embedding import *
from datetime import date

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
  parser.add_argument(
      "--log-dir",
      type=str,
      help="Specifies root directory to store log information.",
      default="./")

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
      nargs="*",
      help=("Specifies the manner in which the provided hypergraph should be "
            "embedded. If multiple are supplied, we train each and them merge "
            "via a linear combination. Options: " + " ".join(
                [o for o in EMBEDDING_OPTIONS])))
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
  parser.add_argument(
      "--embedding-combination-strategy",
      type=str,
      default=COMBINATION_OPTIONS[0],
      help=("If multiple embeddings are specified, how should we combine them?"
            "Options:" + " ".join([o for o in COMBINATION_OPTIONS])))

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
  parser.add_argument(
      "--experiment-rerun",
      type=str,
      help=("If set, supply a path to a previously run experiment."
            "Instead of sampling from the hg, we will just use this one."))
  parser.add_argument(
      "--experiment-shortcut",
      type=str,
      nargs="*",
      help=("Paths to previously constructed embeddings. Used to shortcut the "
            "combine testing process."))
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
  log_dir_path = Path(args.log_dir)
  assert log_dir_path.is_dir()

  def get_uniq_log_name():
    day = str(date.today())
    hypergraph_name = Path(args.hypergraph).stem
    emb_name = ""
    if args.embedding_method is not None:
      if len(args.embedding_method) == 1:
        emb_name = args.embedding_method[0]
      else:
        emb_name = "_".join(args.embedding_method)

    for log_count in range(1, 200):
      name = "{d}.{h}.{e}.{c}.log".format(
          d=day, h=hypergraph_name, e=emb_name, c=log_count)
      path = log_dir_path.joinpath(name)
      if not path.exists():
        return str(path)
    print("Failed to generate log file!", file=sys.stderr)
    exit(1)

  if args.experiment_result is not None:
    log_path = str(
        log_dir_path.joinpath("{d}.{e}.log".format(
            d=str(date.today()), e=Path(args.experiment_result).stem)))
  else:
    log_path = get_uniq_log_name()

  handler = logging.FileHandler(log_path)
  handler.setFormatter(formatter)
  log.addHandler(handler)
  log.info("Logging to %s", log_path)


def PrepLinkPredictionExperiment(hypergraph, args):
  """
  Given data from command line arguments, create a subgraph by removing
  random node-edge connections, embed that hypergraph, and return a list
  of node-edge connections consisting of the removed and negative sampled
  connections. Output is stored in a LinkPredictionData namedtuple
  """
  def load_experiment(path):
    log.info("Recovering information from %s", path)
    exp = ExperimentalResult()
    with open(path, 'rb') as proto:
      exp.ParseFromString(proto.read())
    log.info("Checking Experimental Result")
    assert len(exp.metrics) > 0
    assert len(exp.metrics[0].records) > 0
    assert exp.HasField("hypergraph")

    new_graph = exp.hypergraph
    good_links = [
        (r.node_idx, r.edge_idx) for r in exp.metrics[0].records if r.label
    ]
    bad_links = [
        (r.node_idx, r.edge_idx) for r in exp.metrics[0].records if not r.label
    ]
    return (new_graph, good_links, bad_links, exp.embedding)

  method_name2embedding = {}

  if args.experiment_shortcut is not None:
    log.info("Checking that all shortcuts come from the same experiment")
    new_graph, good_links, bad_links, embedding = load_experiment(args.experiment_shortcut[0])
    method_name2embedding[embedding.method_name] = embedding

    for path in args.experiment_shortcut[1:]:
      tmp_graph, tmp_good_links, tmp_bad_links, embedding = load_experiment(path)
      assert tmp_graph == new_graph
      assert tmp_good_links == good_links
      assert tmp_bad_links == bad_links
      assert embedding.method_name not in method_name2embedding
      method_name2embedding[embedding.method_name] = embedding

    log.info("Loaded shortcuts for:")
    for method in method_name2embedding:
      log.info("\t>\t%s", method)

  elif args.experiment_rerun is not None:
    new_graph, good_links, bad_links, _ = load_experiment(args.experiment_rerun)

  else:
    log.info("Checking that --experiment-lp-probabilty is between 0 and 1")
    assert args.experiment_lp_probability >= 0
    assert args.experiment_lp_probability <= 1

    log.info("Creating subgraph with removal prob. %f",
             args.experiment_lp_probability)
    new_graph, good_links = RemoveRandomConnections(
        hypergraph, args.experiment_lp_probability)
    log.info("Removed %i links", len(good_links))

    log.info("Sampling missing links for evaluation")
    bad_links = SampleMissingConnections(hypergraph, len(good_links))
    log.info("Sampled %i links", len(bad_links))

  log.info("Embedding new hypergraph")
  embedding = Embed(args, new_graph, method_name2embedding)

  return LinkPredictionData(
      hypergraph=new_graph,
      embedding=embedding,
      good_links=good_links,
      bad_links=bad_links,
      removal_prob=args.experiment_lp_probability)


if __name__ == "__main__":
  args = ParseArgs()
  ConfigureLogger(args)

  log.info("=" * 80)
  log.info(str(args))
  log.info("=" * 80)

  log.info("Validating input arguments")
  hypergraph_path = Path(args.hypergraph)

  if args.raw_data:
    log.info("Checking for valid raw-data-format")
    assert args.raw_data_format in PARSING_OPTIONS
    if args.raw_data_format == "DL_MAD_GRADES":
      log.info("Downloading from MADGRADES using supplied API key.")
      raw_data_path = args.raw_data
    else:
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
    log.info("Checking for valid embedding-method, and no duplicates")
    assert len(args.embedding_method) >= 1
    seen_methods = set()
    for method in args.embedding_method:
      assert method in EMBEDDING_OPTIONS
      assert method not in seen_methods
      seen_methods.add(method)
    log.info("Checking for positive dimension")
    assert args.embedding_dimension > 0
    log.info("Checking its safe to write embedding")
    assert not embedding_path.exists()
    assert embedding_path.parent.is_dir()

  log.info("Checking for legal combination strategy")
  assert args.embedding_combination_strategy in COMBINATION_OPTIONS

  log.info("Checking that experimental flags are appropriate")
  log.info("Checking that if experiment is set, experiment-result is too")
  assert bool(args.experiment) == bool(args.experiment_result)
  if args.experiment is not None:
    log.info("Checking that embedding is also specified")
    assert len(args.embedding_method) >= 1
    for method in args.embedding_method:
      assert method in EMBEDDING_OPTIONS
    log.info("Checking for --experiment-result")
    experiment_result_path = Path(args.experiment_result)
    assert not experiment_result_path.exists()
    assert experiment_result_path.parent.is_dir()
    log.info("Checking that experiment is valid")
    for experiment in args.experiment:
      assert experiment in EXPERIMENT_OPTIONS
    if args.experiment_rerun is not None:
      log.info("Checking that rerun path is valid.")
      rerun_path = Path(args.experiment_rerun)
      assert rerun_path.is_file()
    if args.experiment_shortcut is not None:
      log.info("Checking shortcuts")
      for shortcut_path in [Path(p) for p in args.experiment_shortcut]:
        assert shortcut_path.is_file()

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
  log.info(str(args))
