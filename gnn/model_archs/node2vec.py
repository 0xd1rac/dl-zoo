from collections import defaultdict
import random 
import numpy as np 
import gensim.models import Word2Vec

def create_alias_table(probs):
    """
    Given a list of probabilities, construct and return the alias table.
    
    Returns:
        alias: np.array of indices
        prob: np.array of probabilities
    """

    K = len(probs)
    alias = np.zeros(K, dtype=np.int32)
    prob = np.zeros(K, dtype=np.float32)

    scaled_probs = np.array(probs) * K
    small = []
    large = []

    for i, sp in enumerate(scaled_probs):
        if sp < 1.0:
            small.append(i)
        else: 
            large.append(i)

    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled_probs[s]
        alias[s] = l
        scaled_probs[l] = scaled_probs[l] - (1.0 - scaled_probs[s])

        if scaled_probs[l] < 1.0:
            small.append(l)
        else:
            large.append(l)
              
    for i in large + small:
        prob[i] = 1.0

    return alias, prob

def alias_sample(alias, prob):
    K = len(alias)
    i = int(np.floor(np.random.rand() * K))
    if np.random.rand() < prob[i]:
        return i 
    else:
        return alias[i]

class Node2Vec:
    def __init__(self, adj_list, emb_dim=128,
                walk_length=80, num_walks=10,
                p=1.0,q=1.0, workers=1,verbose=False):
        """
        Parameters:
            adj_list: Dictionary representing the graph as an adjacency list.
            Example: {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2]}

            emb_dim: Dimension of the embedding vectors
            walk_length: Length of each random walk.
            num_walks: Number of walks to start from each node.
            p: Return parameter (control the likelihood immediately revisting a node)
            q: In-Out parameter (controls the exploration/exploitation)
            workers: Number of worker threads for Word2Vec
            verbose: If True, prints progress messages 
        """
        
        self.adj_list = adj_list
        self.nodes = list(adj_list.keys())
        self.emb_dim = emb_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p 
        self.q = q
        self.workers = workers
        self.verbose = verbose 

        self.alias_nodes = {}
        self.alias_edges = {}
        
        if self.verbose:
            print("Preprocessing transition probabilities...")
        self._preprocess_transition_probs()

    def _preprocess_transition_probs(self):
        # Precompute alias table for each node (first step)
        for node in self.nodes:
            neighbors = self.adj_list[node]
            if len(neighbors) > 0:
                # Uniform probability for unweighted graphs
                probs = [1.0 / len(neighbors)] * len(neighbors)
                alias, prob = create_alias_table(probs)
                self.alias_nodes[node] = (neighbors, alias, probs)
            
            else:
                self.alias_nodes[node] = ([], None, None)
        
        # Precompute the alias tables for edges 
        for src in self.nodes:
            for dst in self.adj_list[src]:
                self.alias_edges[(src,dst)] = self._get_alias_edge(src, dst)
    

    def _get_alias_edge(self, t, v):
        """
        Compute alias table for edge (t, v) for the transition probability from node 
        t to v and then to a neighbor x of v 
        t -> v -> x
        """
        neighbors = self.adj_list[v]
        probs = []
        for x in neighbors:
            # Determine weight based on distance from t to x
            if x == t:
                weight = 1.0 / self.p

            elif t in self.adj_list[t]:
                weight = 1.0 
            
            else:
                weight = 1.0 / self.q
            
            probs.append(weight)
        
        norm = sum(probs)
        norm_probs = [float(prob) / norm for prob in probs]
        alias, prob = create_alias_table(norm_probs)
        return (neighbors, alias, prob)
    
    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate the random walk of given length starting from start_node.
        """
        walk = [start_node]
        if len(self.adj_list[start_node] == 0):
            return walk 
        
        # First step: choose uniformly from neighbors
        curr = start_node
        neighbors, alias, prob = self.alias_nodes[curr]
        idx = alias_sample(alias, prob)
        next_node = neighbors[idx]
        walk.append(next_node)

        # Subsequent steps: use precomputed alias for edge transitions
        for _ in range(2, walk_length):
            prev = curr
            curr = next_node
            if len(self.adj_list[curr] == 0):
                break
            
            # Use alias table for edge (prev, curr) if available.
            if (prev,curr) in self.alias_edges:
                neighbors, alias, prob = self.alias_edges((prev,curr))
            else:
                # Fall back to uniform sampling if edge data is missing
                neighbors = self.adj_list[curr]
                probs = [1.0 / len(neighbors)] * len(neighbors)
                alias, prob = create_alias_table(probs)
            idx = alias_sample(alias, prob)
            next_node = neighbors[idx]
            walk.append(next_node)
        
        return walk

    def generate_walks(self):
        """
        Generate random walks for all nodes.

        Returns:
            walks: List of walks (each walk is a list of nodes as strings)
        """
        walks = []
        nodes = self.nodes.copy()
        # Each node has self.num_walks random walks
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes: 
                walk = self.node2vec_walk(self.walk_length, node)
                # Convert nodes to strings (required by gensim Word2Vec)
                walk_str= [str(n) for n in walk]
                walks.append(walk_str)
        
        return walks 

    def train(self, window_size=5, min_count=0, sg=1, epochs=1):
        """
        Train embeddings using Word2Vec on generated random walks.
        Returns:
            model: A gensim Word2Vec model with the learned embeddings.
        """
        walks = self.generate_walks()
        if self.verbose:
            print("Training Word2Vec model on generated walks...")

        model = Word2Vec(
            sentences=walks,
            vector_size=self.emb_dim,
            window=window_size,
            min_count=min_count,
            sg=sg,
            workers=self.workers,
            epochs=epochs
        )
        return model