#!/usr/bin/env python3

# This file investigates the output of ExperimentalResult
# proto messages

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import argparse
import sys
from textwrap import dedent
from hypergraph_embedding import *
from pathlib import Path
from statistics import stdev
from scipy.linalg import svd
import numpy as np
import logging
import logging.handlers
from random import sample

log = logging.getLogger()


def ParseArgs():
  parser = argparse.ArgumentParser(
      description="Print results proto information.")
  parser.add_argument(
      "--log-level",
      type=str,
      help=("Specifies level of logging verbosity. "
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
      "-p",
      "--picture",
      type=str,
      help="If set, run SVD on embeddings and create a 2d plot.")
  parser.add_argument(
      "--picture-samples",
      type=int,
      help="Number of embeddings to represent in picture.",
      default=1000)
  parser.add_argument(
      "--bar-chart-path",
      type=str,
      help=("If set, --cumulative and --bar-chart-experiment must also be set."
            "Location to write resulting bar chart with error bounds."))
  parser.add_argument(
      "--bar-chart-experiment",
      type=str,
      help=("If set, --cumulative and --bar-chart-path must also be set."
            "Experiment name to collect bar chart for."))
  parser.add_argument(
      "protos", type=str, help="Path to the results proto buffer", nargs="+")
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


def PrintResult(result, metric, path):
  result_text = dedent("""\
  Path:             {path}
  Hypergraph:       {hypergraph}
  Number of Nodes:  {nodes}
  Number of Edges:  {edges}
  Removal Prob:     {removal}
  --------------------
  Embedding Method: {method}
  Embedding Dim:    {dim}
  --------------------
  Experiment Name:  {e_name}
  Acc:              {e_acc}
  Prec:             {e_pre}
  Recall:           {e_rec}
  F1:               {e_f1}
  Test Set Size:    {e_tss}

  """)
  print(
      result_text.format(
          path=str(path),
          hypergraph=result.hypergraph.name,
          nodes=len(result.hypergraph.node),
          edges=len(result.hypergraph.edge),
          removal=(result.removal_probability
                   if result.HasField("removal_probability") else "N/A"),
          method=result.embedding.method_name,
          dim=result.embedding.dim,
          e_name=metric.experiment_name,
          e_acc=metric.accuracy,
          e_pre=metric.precision,
          e_rec=metric.recall,
          e_f1=metric.f1,
          e_tss=len(metric.records)))


def ExperimentKey(result, metric):
  return "{graph} {experiment} {prob} {method}:{dim}".format(
      graph=result.hypergraph.name,
      experiment=metric.experiment_name,
      prob=result.removal_probability,
      method=result.embedding.method_name,
      dim=result.embedding.dim)


def KeyToMethod(key):
  toks = key.split()
  assert len(toks) >= 4
  method_colon_dim = " ".join(toks[3:])
  toks = method_colon_dim.split(":")
  assert len(toks) == 2
  return toks[0]


def PrintCumulativeResult(key, metrics):
  "Results is an iterable container of EvaluationMetric messages, all related"
  "to the same experiment"

  def TableData(data):
    mean = sum(data) / len(data)
    std = stdev(data) if len(data) > 1 else 0
    return (mean, std, min(data), max(data))

  def TableDataToLine(res):
    return "| {0:5.4f} | {1:5.4f} | {2:5.4f} | {3:5.4f} |".format(*res)

  result_text = dedent("""\
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
          trials=len(metrics),
          acc_line=TableDataToLine(TableData([r.accuracy for r in metrics])),
          pre_line=TableDataToLine(TableData([r.precision for r in metrics])),
          rec_line=TableDataToLine(TableData([r.recall for r in metrics])),
          f1_line=TableDataToLine(TableData([r.f1 for r in metrics]))))


def PrintPicture(result, path, num_samples):
  log.info("Setting up matplotlib")
  fig = plt.figure()
  node_ax = fig.add_subplot(121)
  edge_ax = fig.add_subplot(122)

  def project_2d_samples(idx2emb):
    raw = np.zeros((num_samples, result.embedding.dim), dtype=np.float16)
    for row, idx in enumerate(
        sample(idx2emb.keys(), min(num_samples, len(idx2emb)))):
      raw[row, :] = idx2emb[idx].values
    U, _, _ = svd(raw, full_matrices=0, overwrite_a=True)
    return (U[:, 0], U[:, 1])

  log.info("Projecting nodes into 2d using svd")
  x, y = project_2d_samples(result.embedding.node)
  node_ax.scatter(x, y)
  node_ax.set_title("Nodes")

  log.info("Projecting edges into 2d using svd")
  x, y = project_2d_samples(result.embedding.edge)
  edge_ax.scatter(x, y)
  edge_ax.set_title("Edges")

  log.info("Saving %s", path)
  plt.tight_layout()
  fig.suptitle("{} {}:{}".format(result.hypergraph.name,
                                 result.embedding.method_name,
                                 result.embedding.dim))
  fig.subplots_adjust(top=0.9)
  fig.savefig(path)


def WriteBarChart(bar_chart_path, experiment2metrics, select_exp_name):
  log.info("Selecting accuracies for each method from accumulated data")
  method2vals = {}
  for key, all_metrics in experiment2metrics.items():
    method = KeyToMethod(key)
    if method not in method2vals:
      method2vals[method] = []
    select_vals = [
        m.accuracy for m in all_metrics if m.experiment_name == select_exp_name
    ]
    method2vals[method].extend(select_vals)

  log.info("Checking that we have any metrics for provided experiment type.")
  assert len(method2vals) > 0
  methods = [m for m in method2vals]
  methods.sort(key=lambda m: min(method2vals[m]))
  print(methods)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  for i, m in enumerate(methods):
    ax.bar(
        x=1.2 * i + 1,
        height=sum(method2vals[m]) / len(method2vals[m]),
        width=0.8,
        label=m)
  ax.bar(
      x=[1.2 * i + 1 for i, _ in enumerate(methods)],
      height=[max(method2vals[m]) - min(method2vals[m]) for m in methods],
      width=1,
      bottom=[min(method2vals[m]) for m in methods],
      alpha=0.5,
      color="grey")

  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
  ax.get_xaxis().set_visible(False)

  # Put a legend to the right of the current axis
  lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  title = fig.suptitle(select_exp_name)
  fig.subplots_adjust(top=0.85)
  fig.savefig(
      bar_chart_path, bbox_extra_artists=(lgd, title), bbox_inches='tight')


if __name__ == "__main__":
  args = ParseArgs()
  ConfigureLogger(args)

  log.info("Checking that either print is set, otherwise whats the point?")
  assert args.cumulative or args.individual or args.picture

  log.info("Checking that each provided path exists")
  proto_paths = [Path(proto) for proto in args.protos]
  for path in proto_paths:
    assert path.is_file()

  if args.picture:
    log.info("Checking that the number of embeddings to sample is positive")
    assert args.picture_samples > 0
    picture_path = Path(args.picture)
    log.info("Checking that the picture path is writable")
    assert not picture_path.is_file()
    assert picture_path.parent.is_dir()
    log.info("For now, we are only going to make a picture for a single proto")
    assert len(proto_paths) == 1

  if args.bar_chart_path is not None or args.bar_chart_experiment is not None:
    log.info("Bar chart specified. Checking flags.")
    assert args.cumulative
    assert args.bar_chart_path is not None
    assert args.bar_chart_experiment is not None
    log.info("Checking bar chart path is writable.")
    bar_chart_path = Path(args.bar_chart_path)
    assert not bar_chart_path.exists()
    assert bar_chart_path.parent.is_dir()

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
      for metric in result.metrics:
        if args.individual:
          PrintResult(result, metric, path)
        if args.cumulative:
          key = ExperimentKey(result, metric)
          if key not in experiment2metrics:
            experiment2metrics[key] = []
          experiment2metrics[key].append(metric)

      if args.picture:
        PrintPicture(result, picture_path, args.picture_samples)

  if args.cumulative:
    for key, metric in experiment2metrics.items():
      PrintCumulativeResult(key, metric)
    if args.bar_chart_path is not None:
      WriteBarChart(bar_chart_path, experiment2metrics,
                    args.bar_chart_experiment)
