import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, dense_diff_pool


def to_edge_index(x, adj):
    """
    Convert a dense adjacency matrix (shape [batch_size, N, N]) to an edge_index (shape [2, E]).
    We offset node indices for each graph in the batch by i*N.
    """
    batch_size, N, _ = x.size()  # x has shape [batch_size, N, in_dim], though in_dim not used here
    edge_indices = []
    for i in range(batch_size):
        e_i = adj[i].nonzero(as_tuple=False).T  # shape [2, E_i]
        e_i += i * N
        edge_indices.append(e_i)
    edge_index = torch.cat(edge_indices, dim=1)  # [2, total_E]
    return edge_index


def l2_norm(x, eps=1e-9):
    """
    L2-normalize the rows of x.
    x: [num_nodes, num_features]
    """
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class SAGEBlock(nn.Module):
    """
    A single SAGEConv layer with ReLU + optional L2 norm.
    """
    def __init__(self, in_channels, out_channels, apply_l2_norm=True):
        super().__init__()
        # Use mean aggregator as per the DiffPool paper's default
        self.conv = SAGEConv(in_channels, out_channels, aggr='mean')
        self.apply_l2_norm = apply_l2_norm
    
    def forward(self, x, edge_index):
        """
        Args:
            x (Tensor): [num_nodes, in_channels]
            edge_index (Tensor): [2, num_edges]
        Returns:
            Tensor: [num_nodes, out_channels], after ReLU + L2 norm
        """
        x = self.conv(x, edge_index)
        x = F.relu(x)
        if self.apply_l2_norm:
            x = l2_norm(x)
        return x 


class DiffPoolStep(nn.Module):
    """
    One DiffPool step:
      - An embedding GNN (to produce node embeddings).
      - An assignment GNN (to produce the soft assignment matrix).
      Then calls dense_diff_pool(X_emb, A, S).
    """
    def __init__(self, in_channels, hidden_channels, out_clusters):
        """
        Args:
            in_channels (int): Input feature dim of the nodes before pooling.
            hidden_channels (int): Hidden dim for the GNNs.
            out_clusters (int): Number of clusters to pool to.
        """
        super().__init__()
        # 2-layer GraphSAGE for embeddings
        self.embed_conv1 = SAGEBlock(in_channels, hidden_channels)
        self.embed_conv2 = SAGEBlock(hidden_channels, hidden_channels)

        # 2-layer GraphSAGE for assignments
        self.assign_conv1 = SAGEBlock(in_channels, hidden_channels)
        self.assign_conv2 = SAGEBlock(hidden_channels, out_clusters)
    
    def forward(self, x, adj):
        """
        Args:
            x (Tensor): [batch_size, N, in_channels]
            adj (Tensor): [batch_size, N, N] (dense adjacency)
        Returns:
            h_pooled (Tensor): [batch_size, out_clusters, hidden_channels]
            adj_pooled (Tensor): [batch_size, out_clusters, out_clusters]
            link_loss (Tensor): link prediction loss
            ent_loss (Tensor): entropy loss
        """
        b, N, in_dim = x.size()

        # Flatten x for SAGEConv
        x_flat = x.view(b*N, in_dim)

        # Build edge_index for the entire batch
        edge_index = to_edge_index(x, adj)

        # 1) Embedding GNN
        h_emb = self.embed_conv1(x_flat, edge_index)
        h_emb = self.embed_conv2(h_emb, edge_index)  # final embedding

        # 2) Assignment GNN
        h_assign = self.assign_conv1(x_flat, edge_index)
        h_assign = self.assign_conv2(h_assign, edge_index)
        S = F.softmax(h_assign, dim=-1)  # [b*N, out_clusters]

        # Reshape for dense_diff_pool
        h_emb = h_emb.view(b, N, -1)  # [b, N, hidden_channels]
        S = S.view(b, N, -1)          # [b, N, out_clusters]

        # 3) Perform pooling
        h_pooled, adj_pooled, link_loss, ent_loss = dense_diff_pool(h_emb, adj, S)
        return h_pooled, adj_pooled, link_loss, ent_loss


class DifPoolNet(nn.Module):
    """
    DiffPool Network with:
      1) 2 GraphSAGE layers before first DiffPool
      2) DiffPool layer #1
      3) 3 GraphSAGE layers
      4) DiffPool layer #2
      5) 2 GraphSAGE layers
      6) Final readout (classification)
    """
    def __init__(self, in_channels, hidden_channels, num_clusters1, num_clusters2, num_classes):
        super().__init__()

        # --- GraphSAGE layers BEFORE first DiffPool ---
        self.gnn1 = SAGEBlock(in_channels, hidden_channels)
        self.gnn2 = SAGEBlock(hidden_channels, hidden_channels)

        # --- First DiffPool Step ---
        self.diffpool1 = DiffPoolStep(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_clusters=num_clusters1
        )
        
        # --- GraphSAGE layers AFTER first DiffPool (3 layers) ---
        self.gnn3 = SAGEBlock(hidden_channels, hidden_channels)
        self.gnn4 = SAGEBlock(hidden_channels, hidden_channels)
        self.gnn5 = SAGEBlock(hidden_channels, hidden_channels)

        # --- Second DiffPool Step ---
        self.diffpool2 = DiffPoolStep(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_clusters=num_clusters2
        )

        # --- GraphSAGE layers AFTER second DiffPool (2 layers) ---
        self.gnn6 = SAGEBlock(hidden_channels, hidden_channels)
        self.gnn7 = SAGEBlock(hidden_channels, hidden_channels)

        # --- Final Readout (classification) ---
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, h0, adj0):
        """
        Args:
            h0 (Tensor): [batch_size, N, in_channels] (node features)
            adj0 (Tensor): [batch_size, N, N] (dense adjacency)
        Returns:
            out (Tensor): [batch_size, num_classes] (log probabilities)
            link_loss_total (Tensor)
            ent_loss_total (Tensor)
        """
        batch_size, N, in_dim = h0.size()

        # ---------------------------------------------------------
        # 1) GNN layers before first DiffPool
        # ---------------------------------------------------------
        h0_flat = h0.view(batch_size*N, in_dim)
        edge_index_0 = to_edge_index(h0, adj0)

        # Pass through 2 GraphSAGE layers
        h1 = self.gnn1(h0_flat, edge_index_0)
        h1 = self.gnn2(h1, edge_index_0)

        # Reshape back to [b, N, hidden_channels]
        h1 = h1.view(batch_size, N, -1)

        # ---------------------------------------------------------
        # 2) First DiffPool
        # ---------------------------------------------------------
        h1_pool, adj1_pool, link_loss1, ent_loss1 = self.diffpool1(h1, adj0)
        # h1_pool: [b, num_clusters1, hidden_channels]
        # adj1_pool: [b, num_clusters1, num_clusters1]

        # ---------------------------------------------------------
        # 3) GNN layers after first DiffPool (3 layers)
        # ---------------------------------------------------------
        b1, N1, hdim = h1_pool.size()
        h1_pool_flat = h1_pool.view(b1*N1, hdim)
        edge_index_1 = to_edge_index(h1_pool, adj1_pool)

        h2 = self.gnn3(h1_pool_flat, edge_index_1)
        h2 = self.gnn4(h2, edge_index_1)
        h2 = self.gnn5(h2, edge_index_1)

        h2 = h2.view(b1, N1, -1)

        # ---------------------------------------------------------
        # 4) Second DiffPool
        # ---------------------------------------------------------
        h2_pool, adj2_pool, link_loss2, ent_loss2 = self.diffpool2(h2, adj1_pool)
        # h2_pool: [b, num_clusters2, hidden_channels]
        # adj2_pool: [b, num_clusters2, num_clusters2]

        # ---------------------------------------------------------
        # 5) GNN layers after second DiffPool (2 layers)
        # ---------------------------------------------------------
        b2, N2, hdim2 = h2_pool.size()
        h2_pool_flat = h2_pool.view(b2*N2, hdim2)
        edge_index_2 = to_edge_index(h2_pool, adj2_pool)

        h3 = self.gnn6(h2_pool_flat, edge_index_2)
        h3 = self.gnn7(h3, edge_index_2)
        h3 = h3.view(b2, N2, -1)

        # ---------------------------------------------------------
        # 6) Final Readout
        # ---------------------------------------------------------
        # Mean-pool across the remaining nodes
        h_final = h3.mean(dim=1)  # [b2, hidden_channels]

        out = F.relu(self.lin1(h_final))
        out = self.lin2(out)
        out = F.log_softmax(out, dim=-1)  # [batch_size, num_classes]

        # Combine the link and ent losses from both DiffPool steps
        link_loss_total = link_loss1 + link_loss2
        ent_loss_total = ent_loss1 + ent_loss2

        return out, link_loss_total, ent_loss_total
