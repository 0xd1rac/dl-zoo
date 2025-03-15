import torch 
import torch.nn as nn 
from torch import Tensor
import torch.nn.functional as F
class GraphConvolutionLayer(nn.Module):
    def __init__(self, 
                 in_features:int, out_features:int, bias:bool=True):
        """
        Graph Convolutional Layer as defined in Kipf & Welling (2017).
        
        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node.
            bias (bool): Whether to include bias.
        """
       
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # weight matrix for transformation
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        # Bias term is optional 
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, X: Tensor, adj: Tensor):
        """
        Forward pass of the GCN layer.
        
        Args:
        - X: Node feature matrix (N x C), where N is number of nodes, C is input feature size.
        - adj: Adjacency matrix (N x N), expected to include self-loops.
        
        Returns:
        - Updated node features (N x F), where F is the output feature size.
        """
        I = torch.eye(adj.size(0), device=adj.device) #Identity matrix 
        adj = adj + I # adding self loops 

        # Compute the degree matrix (diagonal matrix with node degrees)
        D = torch.diag(torch.sum(adj, dim=1))

        # Compute the normalized adjacency matrix: D^(-1/2) * A * D^(-1/2)
        D_inv_sqrt = torch.linalg.inv(torch.sqrt(D))
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt  # Normalized adjacency

        # Apply GNC operationrs: H = σ(ÂXW + b)
        output = adj_norm @ X @ self.weights

        if self.bias is not None:
            output += self.bias 
        
        return F.relu(output)
        
