from collections import defaultdict
import random 
import numpy as np 
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch 

def create_alias_table(probs: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of probabilities, construct and return the alias table.
    
    Returns:
        alias: np.array of indices
        prob: np.array of probabilities
    """
    K: int = len(probs)
    alias_table: np.ndarray = np.zeros(K, dtype=np.int32)  # initialized to zero and will store alias indices
    prob_table: np.ndarray = np.zeros(K, dtype=np.float32)  # initialized to zeros and will store the adjusted probabilities

    scaled_probs: np.ndarray = np.array(probs) * K

    # Separating indices into small and large
    small_prob_idx_table: List[int] = []
    large_prob_idx_table: List[int] = []

    for idx, scaled_prob in enumerate(scaled_probs):
        if scaled_prob < 1.0:
            small_prob_idx_table.append(idx)
        else: 
            large_prob_idx_table.append(idx)

    # Constructing the alias table 
    while small_prob_idx_table and large_prob_idx_table:
        small_prob_idx: int = small_prob_idx_table.pop()
        large_prob_idx: int = large_prob_idx_table.pop()
        prob_table[small_prob_idx] = scaled_probs[small_prob_idx]
        alias_table[small_prob_idx] = large_prob_idx
        scaled_probs[large_prob_idx] = scaled_probs[large_prob_idx] - (1.0 - scaled_probs[small_prob_idx])

        if scaled_probs[large_prob_idx] < 1.0:
            small_prob_idx_table.append(large_prob_idx)
        else:
            large_prob_idx_table.append(large_prob_idx)
              
    for i in large_prob_idx_table + small_prob_idx_table:
        prob_table[i] = 1.0

    return alias_table, prob_table

def alias_sample(alias: np.ndarray, prob: np.ndarray) -> int:
    K: int = len(alias)
    i: int = int(np.floor(np.random.rand() * K))
    if np.random.rand() < prob[i]:
        return i 
    else:
        return alias[i]


# (Assume create_alias_table, alias_sample, and Node2Vec class definition are defined as in your previous code.)

class Node2Vec:
    def __init__(self, 
                 adj_list: Dict[int, List[int]], 
                 emb_dim: int = 128, 
                 walk_length: int = 80, 
                 num_walks: int = 10, 
                 p: float = 1.0, 
                 q: float = 1.0, 
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ) -> None:
        """
        Parameters:
            adj_list (Dict[int, List[int]]): Graph as an adjacency list.
                Example: {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2]}
            emb_dim (int): Embedding dimensionality.
            walk_length (int): Length of each random walk.
            num_walks (int): Number of walks per node.
            p (float): Return parameter (controls likelihood of revisiting a node).
            q (float): In-Out parameter (controls exploration/exploitation).
        """
        self.adj_list: Dict[int, List[int]] = adj_list
        self.nodes: List[int] = list(adj_list.keys())
        self.emb_dim: int = emb_dim
        self.walk_length: int = walk_length
        self.num_walks: int = num_walks
        self.device = device
        self.p: float = p 
        self.q: float = q
        self.alias_nodes: Dict[int, Tuple[List[int], np.ndarray, List[float]]] = {}
        self.alias_edges: Dict[Tuple[int, int], Tuple[List[int], np.ndarray, np.ndarray]] = {}
        
        self._preprocess_transition_probs()

    def _preprocess_transition_probs(self) -> None:
        for node in self.nodes:
            neighbors: List[int] = self.adj_list[node]
            if len(neighbors) > 0:
                probs: List[float] = [1.0 / len(neighbors)] * len(neighbors)
                alias, prob = create_alias_table(probs)
                self.alias_nodes[node] = (neighbors, alias, probs)
            else:
                self.alias_nodes[node] = ([], None, None)
        
        for src in self.nodes:
            for dst in self.adj_list[src]:
                self.alias_edges[(src, dst)] = self._get_alias_edge(src, dst)
    
    def _get_alias_edge(self, t: int, v: int) -> Tuple[List[int], np.ndarray, np.ndarray]:
        neighbors: List[int] = self.adj_list[v]
        probs: List[float] = []
        for x in neighbors:
            if x == t:
                weight: float = 1.0 / self.p
            elif t in self.adj_list[t]:
                weight = 1.0 
            else:
                weight = 1.0 / self.q
            probs.append(weight)
        
        norm: float = sum(probs)
        norm_probs: List[float] = [float(prob) / norm for prob in probs]
        alias, prob = create_alias_table(norm_probs)
        return (neighbors, alias, prob)
    
    def node2vec_walk(self, walk_length: int, start_node: int) -> List[int]:
        walk: List[int] = [start_node]
        if len(self.adj_list[start_node]) == 0:
            return walk 
        
        curr: int = start_node
        neighbors, alias, prob = self.alias_nodes[curr]
        idx: int = alias_sample(alias, prob)
        next_node: int = neighbors[idx]
        walk.append(next_node)

        for _ in range(2, walk_length):
            prev: int = curr
            curr = next_node
            if len(self.adj_list[curr]) == 0:
                break
            if (prev, curr) in self.alias_edges:
                neighbors, alias, prob = self.alias_edges[(prev, curr)]
            else:
                neighbors = self.adj_list[curr]
                probs = [1.0 / len(neighbors)] * len(neighbors)
                alias, prob = create_alias_table(probs)
            idx = alias_sample(alias, prob)
            next_node = neighbors[idx]
            walk.append(next_node)
        
        return walk

    def generate_walks(self) -> List[List[str]]:
        walks: List[List[str]] = []
        nodes: List[int] = self.nodes.copy()
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes: 
                walk: List[int] = self.node2vec_walk(self.walk_length, node)
                walk_str: List[str] = [str(n) for n in walk]
                walks.append(walk_str)
        return walks 


    def train_with_hf(self, hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Tuple[SentenceTransformer, Dict[str, torch.Tensor]]:
        """
        Use a pre-trained Hugging Face SentenceTransformer model to obtain node-level embeddings.
        For each random walk, we obtain token embeddings (one per node in the walk) using a separately loaded 
        Hugging Face model. We then aggregate the embeddings of each node over all walks (by averaging) 
        to obtain a final node embedding.

        Parameters:
            hf_model_name (str): Name of the pre-trained Hugging Face model.
            
        Returns:
            model: A SentenceTransformer model.
            node_embeddings: A dictionary mapping node IDs (as strings) to their aggregated embedding (torch.Tensor).
        """
        # Generate random walks; each walk is a list of node strings.
        walks: List[List[str]] = self.generate_walks()
    
        # Load a pre-trained SentenceTransformer model.
        model: SentenceTransformer = SentenceTransformer(hf_model_name, device="cpu")
        # Also load the tokenizer and underlying Hugging Face model separately.
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        auto_model = AutoModel.from_pretrained(hf_model_name)
        auto_model.config.output_hidden_states = True  # Enable hidden states.
        
        node_emb_dict: Dict[str, List[torch.Tensor]] = {}

        for walk in walks:
            # Tokenize the walk; is_split_into_words=True makes sure each node is a separate token.
            encoded = tokenizer(walk, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                # Pass the tokenized input through the model.
                outputs = auto_model(**encoded, output_hidden_states=True, return_dict=True)
            # Retrieve the last hidden state: shape (1, seq_len, hidden_dim)
            last_hidden_state = outputs["hidden_states"][-1]

            # Iterate through each token (node) in the walk.
            for idx, token in enumerate(walk):
                token_emb = last_hidden_state[0, idx, :]  # Shape: (hidden_dim,)
                if token not in node_emb_dict:
                    node_emb_dict[token] = []
                node_emb_dict[token].append(token_emb)

        # Aggregate embeddings for each node (e.g., by averaging).
        final_node_embeddings: Dict[str, torch.Tensor] = {}
        for node, emb_list in node_emb_dict.items():
            final_node_embeddings[node] = torch.mean(torch.stack(emb_list), dim=0)
        
        return model, final_node_embeddings
