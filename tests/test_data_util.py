import unittest
from hypergraph_embedding import Hypergraph
from hypergraph_embedding.data_util import *
from io import StringIO

XML_FORMAT = (
    '<?xml version="1.0" ?>'
    '<!DOCTYPE dblp SYSTEM "dblp.dtd">'
    '<dblp>{}</dblp>')


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


class TestDataUtil(unittest.TestCase):

  def test_ParseDblpXml_one_paper(self):
    _input = make_xml({"T": ["A", "B"]})
    actual = ParseDblpXml(StringIO(_input))

    expected = Hypergraph()
    expected.node[0].name = "A"
    expected.node[0].edges.append(0)
    expected.node[1].name = "B"
    expected.node[1].edges.append(0)
    expected.edge[0].name = "T"
    expected.edge[0].nodes.append(0)
    expected.edge[0].nodes.append(1)

    self.assertEqual(actual, expected)


class TestAMinerUtil(unittest.TestCase):

  def test_small(self):
    _input = ("#* T\n" "#@ A;B")
    actual = [p for p in ParseAMiner(StringIO(_input))]
    expected = [Paper(title="T", authors=["A", "B"])]
    self.assertEqual(actual, expected)

  def test_large(self):
    "AMiner parsing needs to ignore all irrelevant fields"
    DATA_PATH = "test_data/aminer_example.txt"
    with open(DATA_PATH) as ifile:
      actual = [p for p in ParseAMiner(ifile)]
    expected = [
        Paper(
            "ArnetMiner: extraction and mining of academic social networks",
            [
                "Jie Tang",
                "Jing Zhang",
                "Limin Yao",
                "Juanzi Li",
                "Li Zhang",
                "Zhong Su"
            ])
    ]
    self.assertEqual(actual, expected)

  def test_multiple(self):
    "AMiner parsing needs to handle multiple papers properly"
    _input = ("#* T\n" "#@ A;B\n" "#* T2\n" "#@ C;D\n")
    actual = [p for p in ParseAMiner(StringIO(_input))]
    expected = [
        Paper(title="T",
              authors=["A",
                       "B"]),
        Paper(title="T2",
              authors=["C",
                       "D"])
    ]
    self.assertEqual(actual, expected)
