# This file is responsible for preparing experimental data

from . import Hypergraph
from .hypergraph_util import *
from lxml import etree


def ParseDblpXml(xml_source):
  """
    Parses the DBLP xml dump into a co-authorship hypergraph.
    Expects data from dblp.uni-trier.de
    Parsing info: https://dblp.uni-trier.de/faq/How+to+parse+dblp+xml.html
  """
  utf8_parser = etree.XMLParser(encoding='utf-8')
  tree = etree.parse(xml_source, parser=utf8_parser)
  result = Hypergraph()
  author_to_idx = {}
  paper_to_idx = {}
