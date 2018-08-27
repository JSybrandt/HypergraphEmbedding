import unittest
from hypergraph_embedding import Hypergraph
from hypergraph_embedding.data_util import *
from textwrap import dedent

XML_FORMAT = (
    '<?xml version="1.0" encoding="ISO-8859-1"?>'
    '<!DOCTYPE dblp SYSTEM "dblp.dtd">'
    '<dblp>{}</dblp>'
    '</xml>')


def make_article(paper_name, authors, tag):
  return "".join(["<{}>".format(tag)] +
                 ["<author>{}</author>".format(a) for a in authors] +
                 ["<title>{}</title>".format(paper_name),
                  "</{}>".format(tag)])


def make_xml(paper2authors, tag="article"):
  """
  Creates a small valid xml file with given properies.
  Input:
    paper2authors - dict from string to list(string)
  Output:
    XML string
  """
  return XML_FORMAT.format(
      "".join([
          make_article(paper_name,
                       authors,
                       tag) for paper_name,
          authors in paper2authors.items()
      ]))


class TestHarnessTest(unittest.TestCase):

  def test_make_article(self):
    expected = (
        "<tag>"
        "<author>A</author>"
        "<author>B</author>"
        "<title>T</title>"
        "</tag>")
    self.assertEqual(make_article("T", ["A", "B"], "tag"), expected)

  def test_make_xml(self):
    expected = (
        '<?xml version="1.0" encoding="ISO-8859-1"?>'
        '<!DOCTYPE dblp SYSTEM "dblp.dtd">'
        '<dblp>'
        "<tag>"
        "<author>A</author>"
        "<author>B</author>"
        "<title>T1</title>"
        "</tag>"
        "<tag>"
        "<author>C</author>"
        "<author>D</author>"
        "<title>T2</title>"
        "</tag>"
        '</dblp>'
        '</xml>')
    self.assertEqual(
        make_xml({
            "T1": ["A",
                   "B"],
            "T2": ["C",
                   "D"],
        },
                 "tag"),
        expected)
