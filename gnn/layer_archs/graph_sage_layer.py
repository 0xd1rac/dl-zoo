import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GraphSAGELayer(MessagePassing):
    def __init__(self, in_channels, 
                 out_channels,
                 aggregator="mean"
                 ):
        """
        GraphSAGE convolution layer with Differetn Aggregatrs
        Args:
            in_channels (int): Input feature dimension
            out_channels (int): Output feature dimension
            aggregator (str): "mean", "lstm" or "maxpool"
        """

        super().__init__()
        self.aggregator = aggregator
        self.lin = nn.Linear(in_channels * 2, out_channels)

        if aggregator == "lstm":
            self.lstm = nn.LSTM(in_channels, in_channels, batch_first=True)
        elif aggregator == "maxpool":
            self.pool_lin = nn.Linear(in_channels, in_channels)

    
    def forward(self, x, edge_index):
        """
        Forward pass for GraphSAGE
        Args:
            x (Tensor): Node Features 
            edge_index (Tensor): Edge Connections 
        
        Returns:
            Updated node embeddings
        """
        h_neighbors = self.propagate(edge_index, x=x) # Aggregate neighbor embeddings
        h_v = torch.cat([x, h_neighbors], dim=1) # Concatenate self with neighbor info
        return F.relu(self.lin(h_v)) # Apply transformation and activation

    def message(self, x_j):
        """
        Message function for neighbor aggregation
        Args:
            x_j (Tensor): Features of neighboring nodes.
        Returns:
            Aggregated neighbor features 
        """
        if self.aggregator == "mean":
            return x_j #mean aggregaotr just passes values
        elif self.aggregator == "lstm":
            x_j = x_j.view(1, -1, x_j.shape[-1]) # Reshape for LSTM
            h_lstm, _ = self.lstm(x_j) # Apply LSTM aggregation
            return h_lstm.squeeze(0).mean(dim=0) # Return mean LSTM output
        elif self.aggregator == "maxpool":
            return F.relu(self.pool_lin(x_j)).max(dim=0)[0]  # Apply non-linearity and max-pool
