import torch 
import torch.nn as nn 
import torch.nn.function as F 

class GraphAttentionLayer(nn.Module):
    """
    A single layer of the Graph Attention Network (GAT)
    """
    def __init__(self, in_features, out_features, num_heads=1, droput=0.6, alpha=0.2):
        """
        Args:
        - in_features: Input feature size per node.
        - out_features: Output feature size per node.
        - num_heads: Number of attention heads (for multi-head attention).
        - dropout: Dropout probability.
        - alpha: Negative slope for LeakyReLU.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.alpha = alpha

        # Feature transformation matrix
        self.W = nn.Parameter(torch.Tensor(num_heads, in_features, out_features))

        # Attention weight vector
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * out_features, 1))

        # dropout layer
        self.dropout = nn.Dropout(droput)

        # LeakyReLU 
        self.leakyrelu = nn.LeakyReLU(alpha)

    
    def forward(self, X, adj):
        """
        Forward pass of GAT layer.
        
        Args:
        - X: Node feature matrix (N x in_features).
        - adj: Adjacency matrix (N x N).
        
        Returns:
        - Output feature matrix (N x out_features).
        """
        N = X.shape[0] # num of nodes in the graph
        
        # (1) Apply linear transformation to node features (W * h)
        H = torch.matmul(X, self.W)  # Shape: (num_heads, N, out_features)

        # (2) Compute attention scores 
        # Concatenation trick using broadcasting
        H_repeat = H.unsqueeze(2).expand(-1, -1, N, -1)  # Shape: (num_heads, N, N, out_features)
        H_pairwise = torch.cat([H_repeat, H_repeat.transpose(1, 2)], dim=-1)  # Shape: (num_heads, N, N, 2*out_features)

        # (3) Apply attention mechanism
        e = self.leakyrelu(torch.matmul(H_pairwise, self.a).squeeze(-1))  # Shape: (num_heads, N, N)
        
        # (4) Mask non-existing edges by setting attention scores to -inf
        e = e.masked_fill(adj.unsqueeze(0) == 0, float('-inf'))

         # (5) Apply softmax to compute attention coefficients
        attention = F.softmax(e, dim=-1)  # Shape: (num_heads, N, N)
        attention = self.dropout(attention)  # Apply dropout

        # (6) Compute final node representations
        H_out = torch.matmul(attention, H)  # Shape: (num_heads, N, out_features)

        # (7) Aggregate multi-head outputs (mean-pooling or concatenation)
        if self.num_heads > 1:
            H_out = H_out.mean(dim=0)  # Mean-pooling across heads

        return H_out