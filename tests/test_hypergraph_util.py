import unittest
from hypergraph_embedding import Hypergraph
from hypergraph_embedding.hypergraph_util import *

def EmptyHypergraph():
    return Hypergraph();

class HypergraphUtilFunctions(unittest.TestCase):
    def test_ToSparseMatrix(self):
        self.assertEqual(ToSparseMatrix(EmptyHypergraph()), 1)

if __name__ == "__main__":
    unittest.main()
