# This file is responsible for preparing experimental data
from . import Hypergraph
from .hypergraph_util import *
from collections import namedtuple
import logging

log = logging.getLogger()

global PARSING_OPTIONS


def ParseRawIntoHypergraph(args, raw_data_path):
  log.info("Parsing %s with %s method", raw_data_path, args.raw_data_format)
  with raw_data_path.open('r') as raw_file:
    hypergraph = PARSING_OPTIONS[args.raw_data_format](raw_file)
  if args.name:
    log.info("Setting hypergraph name to %s", args.name)
    log.info("Good name!")
    hypergraph.name = args.name
  else:
    log.info("Setting hypergraph name to %s", args.hypergraph)
    log.info("Bad name :(")
    hypergraph.name = args.hypergraph
  return hypergraph


# Used to store paper data
Paper = namedtuple("Paper", ['title', 'authors'])


def SnapCommunityToHypergraph(snap_source):
  hypergraph = Hypergraph()
  for edge_idx, node_str in enumerate(snap_source):
    for node_idx in node_str.split():
      AddNodeToEdge(hypergraph, int(node_idx), edge_idx)
  return hypergraph


def AMinerToHypergraph(aminer_source):
  return PapersToHypergraph(ParseAMiner(aminer_source))


def ParseAMiner(aminer_source):
  """
    Parses data in AMiner's format.
    Ignores all fields except title and authors
    More information on this format here: https://aminer.org/aminernetwork

    Input:
      - aminer_source : a file-like object
    Output: (yield)
      - a list of Papers (named tuple)
   """
  log.info("Parsing AMiner data")
  last_seen_title = None
  for line in aminer_source:
    if line.startswith("#*"):  # paper title line
      last_seen_title = line[2:].strip()
    elif line.startswith("#@"):  # authors line
      authors = line[2:].strip().split(';')
      yield Paper(title=last_seen_title, authors=authors)


def PapersToHypergraph(parser):
  """
    Converts paper data into hypergraph.
    Input:
      - A iterable type that supplies Paper tuples
    Output:
      - A hypergraph
    """
  log.info("Converting papers to hypergraph")
  title2idx = {}
  author2idx = {}
  result = Hypergraph()
  for paper in parser:
    if paper.title not in title2idx:
      title2idx[paper.title] = len(title2idx)
    for author in paper.authors:
      if author not in author2idx:
        author2idx[author] = len(author2idx)
      AddNodeToEdge(
          result, author2idx[author], title2idx[paper.title], author,
          paper.title)
  return result


PARSING_OPTIONS = {
    "AMINER": AMinerToHypergraph,
    "SNAP": SnapCommunityToHypergraph
}
