# This file is responsible for preparing experimental data

from . import Hypergraph
from .hypergraph_util import *
from lxml import etree
from collections import namedtuple

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

  last_seen_title = None
  for line in aminer_source:
    if line.startswith("#*"):  # paper title line
      last_seen_title = line[2:].strip()
    elif line.startswith("#@"):  # authors line
      authors = line[2:].strip().split(';')
      yield Paper(title=last_seen_title, authors=authors)
