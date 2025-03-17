
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DiffPoolLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_clusters):
        """
        DiffPool Layer: learns a soft clustering assignment and pools the graph into a coarser representation
        Args:
            in_channels(int): Number of input node features 
            hidden_channels(int): Number of hidden node features 
            num_clusters(int): Number of clusters (output nodes after pooling)
        """
        super().__init__()

        # GNN To genreate node embeddings
        self.gnn_embed = GCNConv(in_channels, hidden_channels)

        # GNN to generate soft assignments matrix S
        self.gnn_assign = GCNConv(in_channels, num_clusters)
    
    def forward(self, x, edge_index):
        """
        Forward pass for DiffPool Layer.

        Args:
            x (Tensor): Node features (N x d)
            edge_index (Tensor): Graph edge connections (2 x E)

        Returns:
            x_pooled (Tensor): New pooled node features (K x d)
            edge_index_pooled (Tensor): New edge connections (K x K)
            S (Tensor): Soft assignment matrix (N x K)

        """
        # Step 1: Compute node embeddings (H)
        x_emb = F.relu(self.gnn_embed(x, edge_index))

        # Step 2: Compute Soft Assignment Matrix (S)
        S = self.gnn_assign(x, edge_index)  # Raw scores
        S = F.softmax(S, dim=-1)  # Convert into probability distribution (N x K)
        
        # Step 3: Compute new node features 
        x_pooled = torch.matmul(S.T, x_emb)

        # Step 4: Compute new adjacney matrix (A_pooled)
        A = torch.zeros((x.size(0), x.size(0)), device=x.device)
        A[edge_index[0], edge_index[1]] = 1  # Convert edge list to adjacency matrix

        A_pooled = torch.matmul(torch.matmul(S.T, A), S)  # A^(l+1) = S^T * A * S

        # Convert back to edge list
        edge_index_pooled = A_pooled.nonzero(as_tuple=False).T
        
        return x_pooled, edge_index_pooled, S