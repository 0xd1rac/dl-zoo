
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv, SAGEConv

class DiffPoolLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters, embedding_conv_type="graphsage", cluster_conv_type="graphsage"):
        """
        DiffPool Layer: learns a soft clustering assignment and pools the graph into a coarser representation
        Args:
            input_dim(int): dim of input feature vector
            hidden_dim(int): dim of output feature vector
            num_clusters(int): Number of clusters (output nodes after pooling)
        """
        super().__init__()

        ## GNN to generate node embeddings H^(l)
        if embedding_conv_type == "gcn":
            self.gnn_embed = GCNConv(input_dim, hidden_dim)
        elif embedding_conv_type == "graphsage":
            self.gnn_embed = SAGEConv(input_dim, hidden_dim)
        elif embedding_conv_type == "gat":
            self.gnn_embed = GATConv(input_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported embedding_conv_type: {embedding_conv_type}")
        
        # GNN to generate S matrix 
        if cluster_conv_type == "gcn":
            self.gnn_s = GCNConv(input_dim, num_clusters)
        elif cluster_conv_type == "graphsage":
            self.gnn_s = SAGEConv(input_dim, num_clusters)
        elif cluster_conv_type == "gat":
            self.gnn_s = GATConv(input_dim, num_clusters)
        else:
            raise ValueError(f"Unsupported cluster_conv_type: {cluster_conv_type}")

    def forward(self, h_prev, A):
        """
        Forward pass for DiffPool Layer.

        Args:
            h_prev (Tensor): Node features (N x d)
            A (Tensor): Dense adjacency matrix of shape (N x N)

        Returns:
            h_pooled (Tensor): New pooled node features computed as S^T * h.
            A_pooled (Tensor): Pooled adjacency matrix computed as S^T * A * S.

            S (Tensor): Soft assignment matrix (N x num_clusters)

        """
        # Convert the dense adjacency matrix A to an edge index for use with the GNN layers.
        edge_index = A.nonzero(as_tuple=False).T  # shape (2, E)

        # Step 1: Compute node embeddings (H)
        h = F.relu(self.gnn_embed(h_prev, edge_index))

        # Step 2: Compute Soft Assignment Matrix (S)
        S = self.gnn_s(h, edge_index) # raw scores
        S = F.softmax(S, dim=-1) # Convert into probability distribution (N x K)

         # Step 3: Pool node embeddings: h_pooled = S^T * h.
        h_pooled = torch.matmul(S.T, h)

        # Step 4: Compute new adjacney matrix (A_pooled)
        A_pooled = torch.matmul(torch.matmul(S.T, A),S)
        return h_pooled, A_pooled, S