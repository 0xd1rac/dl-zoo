import torch 
import torch.nn as nn
import random
import numpy as np

# Generator: LSTM-based sequence generator
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lin = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        out, _ = self.lstm(z)
        return self.lin(out)
    

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.lin(out[:, -1, :]))