# This file is responsible for preparing experimental data
from . import Hypergraph
from .hypergraph_util import *
from lxml import etree
from collections import namedtuple
import logging

log = logging.getLogger()

# Used to store paper data
Paper = namedtuple("Paper", ['title', 'authors'])


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
          result,
          author2idx[author],
          title2idx[paper.title],
          author,
          paper.title)
  return result
