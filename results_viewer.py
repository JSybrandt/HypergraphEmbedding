#!/usr/bin/env python3

# This file investigates the output of ExperimentalResult
# proto messages

import argparse
import sys
from textwrap import dedent
from hypergraph_embedding import *
from pathlib import Path
from statistics import stdev


def ParseArgs():
  parser = argparse.ArgumentParser(
      description="Print results proto information.")
  parser.add_argument(
      "protos",
      type=str,
      help="Path to the results proto buffer",
      nargs="+")
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
  return parser.parse_args()


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

  def meanAndStd(data):
    return (sum(data) / len(data), stdev(data))

  m_acc, s_acc = meanAndStd([r.accuracy for r in results])
  m_pre, s_pre = meanAndStd([r.precision for r in results])
  m_rec, s_rec = meanAndStd([r.recall for r in results])
  m_f1, s_f1 = meanAndStd([r.f1 for r in results])

  result_text = dedent(
      """\
  Experiment: {key}
  Trials:     {trials}

  Value     |  MEAN  |  STD   |
  -----------------------------
  Accuracy  | {ma:5.4f} | {sa:5.4f} |
  Precision | {mp:5.4f} | {sp:5.4f} |
  Recall    | {mr:5.4f} | {sr:5.4f} |
  F1        | {mf:5.4f} | {sf:5.4f} |
  -----------------------------
      """)
  print(
      result_text.format(
          key=key,
          trials=len(results),
          ma=m_acc,
          sa=s_acc,
          mp=m_pre,
          sp=s_pre,
          mr=m_rec,
          sr=s_rec,
          mf=m_f1,
          sf=s_f1))


if __name__ == "__main__":
  args = ParseArgs()
  # Checking that either print is set, otherwise whats the point?
  assert args.cumulative or args.individual

  proto_paths = [Path(proto) for proto in args.protos]
  for path in proto_paths:
    assert path.is_file()

  experiment2metrics = {}

  for path in proto_paths:
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
