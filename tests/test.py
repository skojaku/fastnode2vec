import shutil
import unittest

import networkx as nx
import numpy as np
from scipy import sparse

import fastnode2vec


class TestCalc(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(self.G)

    def test_node2vec(self):
        model = fastnode2vec.Node2Vec()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32
    
    def test_node2vec_p_q(self):
        model = fastnode2vec.Node2Vec(p = 2, q = 1)
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32

if __name__ == "__main__":
    unittest.main()
