#!/usr/bin/env python3

# This file investigates the output of ExperimentalResult
# proto messages

import argparse
import sys
from textwrap import dedent
from hypergraph_embedding import *
from pathlib import Path
from statistics import stdev
import logging
import logging.handlers

log = logging.getLogger()


def ParseArgs():
  parser = argparse.ArgumentParser(
      description="Print results proto information.")
  parser.add_argument(
      "--log-level",
      type=str,
      help=(
          "Specifies level of logging verbosity. "
          "Options: CRITICAL, ERROR, WARNING, INFO, DEBUG, NONE"),
      default="INFO")
  parser.add_argument(
      "-i",
      "--individual",
      action="store_true",
      help="If set, print each individual result")
  parser.add_argument(
      "-c",
      "--cumulative",
      action="store_true",
      help="If set, group results by experiment key and print group stats.")
  parser.add_argument(
      "protos",
      type=str,
      help="Path to the results proto buffer",
      nargs="+")
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


def PrintResult(result):
  result_text = dedent(
      """\
  --------------------
  Hypergraph:       {hypergraph}
  Number of Nodes:  {nodes}
  Number of Edges:  {edges}
  --------------------
  Embedding Method: {method}
  Embedding Dim:    {dim}
  --------------------
  {metrics}
  --------------------""")
  print(
      result_text.format(
          hypergraph=result.hypergraph.name,
          nodes=len(result.hypergraph.node),
          edges=len(result.hypergraph.edge),
          method=result.embedding.method_name,
          dim=result.embedding.dim,
          metrics=result.metrics))


def ExperimentKey(result):
  return "{graph} {method}:{dim}".format(
      graph=result.hypergraph.name,
      method=result.embedding.method_name,
      dim=result.embedding.dim)


def PrintCumulativeResult(key, results):
  "Results is an iterable container of EvaluationMetric messages, all related"
  "to the same experiment"

  def TableData(data):
    mean = sum(data) / len(data)
    std = stdev(data) if len(data) > 1 else 0
    return (mean, std, min(data), max(data))

  def TableDataToLine(res):
    return "| {0:5.4f} | {1:5.4f} | {2:5.4f} | {3:5.4f} |".format(*res)

  result_text = dedent(
      """\
  Experiment: {key}
  Trials:     {trials}

  Value     |  MEAN  |  STD   |  MIN   |  MAX   |
  -----------------------------------------------
  Accuracy  {acc_line}
  Precision {pre_line}
  Recall    {rec_line}
  F1        {f1_line}
  -----------------------------------------------
      """)
  print(
      result_text.format(
          key=key,
          trials=len(results),
          acc_line=TableDataToLine(TableData([r.accuracy for r in results])),
          pre_line=TableDataToLine(TableData([r.precision for r in results])),
          rec_line=TableDataToLine(TableData([r.recall for r in results])),
          f1_line=TableDataToLine(TableData([r.f1 for r in results]))))


if __name__ == "__main__":
  args = ParseArgs()
  ConfigureLogger(args)

  log.info("Checking that either print is set, otherwise whats the point?")
  assert args.cumulative or args.individual

  log.info("Checking that each provided path exists")
  proto_paths = [Path(proto) for proto in args.protos]
  for path in proto_paths:
    assert path.is_file()

  experiment2metrics = {}

  for path in proto_paths:
    log.info("Parsing %s", path)
    with path.open('rb') as proto_file:
      result = ExperimentalResult()
      try:
        result.ParseFromString(proto_file.read())
      except:
        print(
            "[ERROR]:{} is not an Experimental Result".format(path),
            file=sys.stderr)
        exit(1)
      if args.individual:
        PrintResult(result)
      if args.cumulative:
        key = ExperimentKey(result)
        if key not in experiment2metrics:
          experiment2metrics[key] = []
        experiment2metrics[key].append(result.metrics)
  if args.cumulative:
    for key, results in experiment2metrics.items():
      PrintCumulativeResult(key, results)
