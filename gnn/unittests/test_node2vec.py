import unittest
import torch 
import numpy as np
from sentence_transformers import SentenceTransformer
from layer_archs.node2vec import Node2Vec, create_alias_table, alias_sample

class TestNode2Vec(unittest.TestCase):
    def setUp(self):
        # Creat a simple graph represented as an adj list.
        self.adj_list = {
            0: [1, 2],
            1: [0, 2, 3],
            2: [0, 1],
            3: [1]
        }

        # Create an instance of Node2Vec with small parameters for quick testing
        self.n2v = Node2Vec(
            adj_list=self.adj_list, 
            emb_dim=16,
            walk_length=10,
            num_walks=5, 
            p=1.0,
            q=1.0
        )

    
    def test_create_alias_table(self):
        # Test that create_alias_table returns arrays of the correct length and valid probabilities.
        probs = [0.2, 0.5, 0.3]
        alias, prob = create_alias_table(probs)
        self.assertEqual(len(alias), len(probs))
        self.assertEqual(len(prob), len(probs))
        self.assertTrue(np.all(prob >= 0))
        self.assertTrue(np.all(prob <= 1))
    
    def test_alias_sample(self):
        # Test that alias_sample returns a valid index.
        alias = np.array([1,0])
        prob = np.array([0.7, 0.3])
        idx = alias_sample(alias, prob)
        self.assertIn(idx, [0,1])

    def test_generate_walks(self):
        # Test that generate_walks returns the expected number of walks and that each walk is a list of strings.
        walks = self.n2v.generate_walks()
        expected_num_walks = self.n2v.num_walks * len(self.n2v.nodes)
        self.assertEqual(len(walks), expected_num_walks)
        for walk in walks:
            self.assertIsInstance(walk, list)

            # In our test graph every node has neighbors so walk length should be as specified.
            self.assertEqual(len(walk), self.n2v.walk_length)

            for node in walk:
                self.assertIsInstance(node, str)


    def test_node2vec_walk(self):
        # Test that a single random walk is generated correctly 
        walk = self.n2v.node2vec_walk(self.n2v.walk_length, 0)
        self.assertIsInstance(walk, list)
        self.assertEqual(len(walk), self.n2v.walk_length)

    def test_train_with_hf_node_embeddings(self):
        # Test the train_with_hf method that returns node-level embeddings
        # This may take a little time as it calls a Huggig Face model
        model, node_embeddings = self.n2v.train_with_hf(hf_model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Check that model is a SentenceTransformer instance.
        self.assertIsInstance(model, SentenceTransformer)

        # Check that node_embeddings is a dictionary.
        self.assertIsInstance(node_embeddings, dict)

         # Verify that each key is a string and each value is a torch.Tensor.
        for node_id, emb in node_embeddings.items():
            self.assertIsInstance(node_id, str)
            self.assertIsInstance(emb, torch.Tensor)
            # Optionally, check the dimension of the embedding.
            self.assertEqual(emb.shape[0] > 0, True)

if __name__ == "__main__":
    unittest.main()