import torch 
import torch.nn as nn 
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNLayer(MessagePassing):
    def __init__(self, in_channels:int, out_channels:int):
        """
        Graph Convolution Layer (GCN)
        Args:
            in_channels(int): Input feature dimension
            out_channels(int): Output feature dimension
        """
        super(GCNLayer, self).__init__(aggr="add") #GCN uses "sum" aggregation
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass for GCN layer
        Args:
            x (Tensor): Node features (N x d)
            edge_index (Tensor): Edge Connections (2 x E)
        
        Returns:
            Updated node embeddings
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index #Extract source and target nodes 

        # Compute degree normalization (D^(-1/2))
        deg = degree(row, x.size(0, dtype=x.dtype))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 # Avoid division by zero

        # normalize features 
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Aggregate neighbor features and apply normalization
        h_neighbors = self.propagate(edge_index, x=x, norm=norm)

        # Apply weight transformation and activation
        h_v = self.lin(h_neighbors)
        return F.relu(h_v)
    
    def message(self, x_j, norm):
        """
        Message function for aggregation 
        Args:
            x_j (Tensor): Features of neighboring nodes
            norm (Tensor): Degree normalization values 

        Returns:
            Aggregated neighbor features 
        """
        return norm.view(-1, 1) * x_j #Apply normalization factor


