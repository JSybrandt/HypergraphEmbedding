#!/usr/bin/env python3

import argparse
from pathlib import Path
from hypergraph_embedding import ExperimentalResult

def ParseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--compare-correct",
      action="store_true",
      help="If set, compare the two results by their correctly predicted links")
  parser.add_argument(
      "--compare-wrong",
      action="store_true",
      help=("If set, compare the two results by their incorrectly predicted "
            "links"))
  parser.add_argument(
      "paths",
      nargs=2,
      help="Must supply two paths to ExperimentalResult proto messages.")

  return parser.parse_args()

def ToInterestingRecords(result, exp_name, take_correct):

  def take_rec(r):
    if take_correct:
      return r.label == r.prediction
    else:
      return r.label != r.prediction

  metric = [m for m in result.metrics if m.experiment_name == exp_name][0]
  return set((r.node_idx, r.edge_idx)
          for r in metric.records
          if take_rec(r))

if __name__ == "__main__":
  args = ParseArgs()
  res1_path, res2_path = [Path(p) for p in args.paths]
  # must have supplied real files
  assert res1_path.is_file()
  assert res2_path.is_file()
  # must have set one but not both
  assert args.compare_correct or args.compare_wrong
  assert not (args.compare_correct and args.compare_wrong)

  result1 = ExperimentalResult()
  result2 = ExperimentalResult()

  with res1_path.open("rb") as proto:
    result1.ParseFromString(proto.read())
  with res2_path.open("rb") as proto:
    result2.ParseFromString(proto.read())

  shared_experiments = set(m.experiment_name
                           for m in result1.metrics).intersection(
                               set(m.experiment_name
                                   for m in result2.metrics))
  assert len(shared_experiments) > 0

  for name in shared_experiments:
    recs1 = ToInterestingRecords(result1, name, args.compare_correct)
    recs2 = ToInterestingRecords(result2, name, args.compare_correct)
    jac = len(recs1.intersection(recs2)) / len(recs1.union(recs2))

    print(name, jac)
