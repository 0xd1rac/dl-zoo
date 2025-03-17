
import torch 
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels:int, hidden_channels:int, out_channels:int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        self.relu = F.relu()
    
    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x,edge_index))
        mu = self.conv_mu(x, edge_index)
        log_std = self.conv_logstd(x, edge_index)
        return mu, log_std
    
# Decoder: reconstructs the graph structure using the inner product 
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z):
        adj_recon = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_recon

# Generator: maps noise from a prior distribution to the latent space
class Generator(nn.Module):
    def __init__(self, noise_dim:int, hidden_dim:int, latent_dim:int):
        super().__init__()
        self.fc1 = nn.Linear(noise_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, noise):
        x = F.relu(self.fc1(noise))
        x = F.relu(self.fc2(x))
        latent = self.fc3(x)
        return latent

# Discriminator: distinguishes between encoded (real) latent codes and generated (fake) latent codes
class Discriminator(nn.Module):
    def __init__(self, latent_dim:int, hidden_dim:int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
