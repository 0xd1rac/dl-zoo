import unittest
import random
import numpy as np
from gensim.models import Word2Vec
from gnn.layer_archs.node2vec import *

random.seed(42)
np.random.seed(42)

class TestNode2Vec(unittest.TestCase):
    def setUp(self):
        # Example unweighted graph represented as an adjacency list.
        self.example_adj_list = {
                                0: [1, 2],
                                1: [0, 2],
                                2: [0, 1, 3],
                                3: [2]
                            }
        # Create an instance of Node2Vec with a small graph
        self.n2v = Node2Vec(adj_list=self.example_adj_list, 
                            emb_dim=8, 
                            walk_length=5, 
                            num_walks=2, 
                            p=0.5, q=0.5, workers=3, verbose=True)
        
    def test_generate_walks(self):
        # Generate walks and check that each walk has the expected length and valid nodes.
        walks = self.n2v.generate_walks()
        self.assertEqual(len(walks), 8)
        for walk in walks:
            # Walk length may be shorter than walk_length if a dead end is reached. 
            # so check that the length is at most work_length and at least 1 
            self.assertGreaterEqual(len(walk), 1)
            self.assertLessEqual(len(walk), self.n2v.walk_length)

            # Check that every node in the walk is one of the graph nodes (as string)
            for node in walk:
                self.assertIn(node, [str(n) for n in self.example_adj_list.keys()])

    def test_train_embeddings(self):
        # Train the model using generated walks 
        model = self.n2v.train(window_size=3, epochs=3)

        # Check that model returns an embedding for each node in our graph
        for node in self.example_adj_list.keys():
            # gensim Word2Vec uses string keys
            self.assertIn(str(node), model.wv)

            # Embedding dim should match self.n2v.emb_dim
            embedding = model.wv[str(node)]
            self.assertEqual(len(embedding), self.n2v.emb_dim)

if __name__ == '__main__':
    unittest.main()

