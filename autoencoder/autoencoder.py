import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Autoencoder(nn.Module):
    def __init__(self, 
                 input_channels: int, 
                 hidden_dim_1: int, 
                 hidden_dim_2: int,
                 latent_dim: int):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim_1, hidden_dim_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim_2, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Decoder 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim_2, hidden_dim_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim_1, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() # Normalize to 0-1
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
