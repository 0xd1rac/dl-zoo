import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_channels:int, latent_dim:int):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(8192, latent_dim)
        self.fc_log_var = nn.Linear(8192, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        print(f"x.shape: {x.shape}")
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim:int, output_channels:int):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.de_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.de_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.de_conv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # Added intermediate layer
        self.de_conv4 = nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1)  # Final layer

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.de_conv1(x))  # -> [-1, 64, 8, 8] -1 infers the batch_size
        x = F.relu(self.de_conv2(x))  # -> [-1, 32, 16, 16]
        x = F.relu(self.de_conv3(x))  # -> [-1, 16, 32, 32]
        x_reconstructed = torch.sigmoid(self.de_conv4(x))  # -> [-1, output_channels, 64, 64]
        return x_reconstructed

class VAE(nn.Module):
    def __init__(self, input_channels:int, latent_dim:int):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)
    
    def reparameterize(self, mu:torch.Tensor, log_var:torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var

def vae_loss(reconstructed_x:torch.Tensor, x:torch.Tensor, mu:torch.Tensor, log_var:torch.Tensor) -> torch.Tensor:
    reconstruction_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence
