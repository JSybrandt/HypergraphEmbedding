import unittest
from hypergraph_embedding import Hypergraph
from hypergraph_embedding.data_util import *
from io import StringIO


class TestAMinerUtil(unittest.TestCase):

  def test_small(self):
    _input = ("#* T\n#@ A;B")
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
    _input = ("#* T\n#@ A;B\n#* T2\n#@ C;D\n")
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


class TestPapersToHypergraph(unittest.TestCase):

  def test_small(self):
    "Should load one edge. Indices created in order of occurance."
    _input = [Paper("T", ["A", "B"])]
    actual = PapersToHypergraph(_input)
    expected = Hypergraph()
    AddNodeToEdge(expected, 0, 0, "A", "T")
    AddNodeToEdge(expected, 1, 0, "B", "T")
    self.assertEqual(actual, expected)

  def test_generator_small(self):
    raw_text = ("#* T\n" "#@ A;B\n")
    _input = ParseAMiner(StringIO(raw_text))
    actual = PapersToHypergraph(_input)
    expected = Hypergraph()
    AddNodeToEdge(expected, 0, 0, "A", "T")
    AddNodeToEdge(expected, 1, 0, "B", "T")
    self.assertEqual(actual, expected)

  def test_typical(self):
    "PapersToHypergraph should handle multiple authors / papers"
    _input = [Paper("X", ["A", "B"]), Paper("Y", ["A", "C"])]
    actual = PapersToHypergraph(_input)
    expected = Hypergraph()
    AddNodeToEdge(expected, 0, 0, "A", "X")
    AddNodeToEdge(expected, 1, 0, "B", "X")
    AddNodeToEdge(expected, 0, 1, "A", "Y")
    AddNodeToEdge(expected, 2, 1, "C", "Y")
    self.assertEqual(actual, expected)
