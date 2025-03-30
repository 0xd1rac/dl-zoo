import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax

class GATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.6):
        """
        Graph Attention Layer (GAT)
        Args:
            in_channels (int): Input feature dimension
            out_channels (int): Output feature dimension (per head)
            heads(int): Number of attention heads
            dropout (float): Dropout proba
        
        """
        super(GATLayer, self).__init__(aggr="add") # Aggregation method
        self.heads = heads 
        self.dropout = dropout

        # Learnable weight matrices for transformation
        self.lin = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.attn = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels)) # Attention parametr
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize parameters using Xavier initialization
        """
        nn.init.xavier_uniform(self.lin.weight)
        nn.init.xavier_uniform(self.attn)

    def forward(self, x, edge_index):
        """
        Forward pass for GAT layer
        Args:
            x (Tensor): Node features (N x d)
            edge_index (Tensor): Edge list (2 x E)
        Returns:
            Updated node embeddings
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        x = x.view(-1, self.heads, x.shape[-1] // self.heads) # Split into attention heads

        return self.propagate(edge_index, x=x)
    
    def message(self, x_j, x_i, index, ptr, size_i):
        """
        Message function for attention-based aggregation
        Args:
            (j) -> (i)

            x_j (Tensor): Features of source nodes
            x_i (Tensor): Features of target nodes
            
            index (Tensor): Edge index for attention calculation
            ptr, size_i: help with segment-wise softmax (for efficiency)

        Returns:
            Attention-weighted features
        """
         # Compute attention scores
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.attn).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)  # Apply LeakyReLU activation
        alpha = softmax(alpha, index, ptr, size_i)  # Softmax over neighbors
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # Apply dropout
        return x_j * alpha.unsqueeze(-1)  # Multiply by attention scores
    
    def aggregate(self, inputs, index, dim_size=None):
        """
        Aggregate function for summing up neighbor messages
        """
        return torch.sum(inputs, dim=1)  # Summing up messages per node