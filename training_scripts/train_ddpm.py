
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
from diffusion.models.ddpm import Unet, LinearNoiseScheduler
from utils.datamanger import DataManager

if __name__ == "__main__":
    # Hyperparameters
    epochs = 20
    batch_size = 128
    learning_rate = 1e-4
    num_timesteps = 1000  # Total diffusion steps
    noise_schedule_type = "linear"  # Change to "cosine" to experiment with cosine schedule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beta_start = 0.0001
    beta_end = 0.02
    noise_scheduler = LinearNoiseScheduler(num_timesteps, beta_start, beta_end)

    train_loader, test_loader = DataManager.get_mnist_dataloaders()

    # Define the DDPM model configuration (for MNIST: 1-channel images).
    model_config = {
        "im_channels": 1,
        "down_channels": [64, 128, 256, 256],
        "mid_channels": [256, 256, 128],
        "time_emb_dim": 128,
        "down_sample": [True, True, False],
        "num_down_layers": 2,
        "num_mid_layers": 1,
        "num_up_layers": 2
    }

    # Instantiate the model (make sure your Unet model code is available).
    model = Unet(model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            current_batch = images.shape[0]

            # Sample random time steps for each image.
            t = torch.randint(0, num_timesteps, (current_batch,), device=device).long()

            # Sample random noise with the same shape as images.
            noise = torch.randn_like(images)

            # Create noisy images using the scheduler's add_noise function.
            noisy_images = noise_scheduler.add_noise(images, noise, t)

            # predict noise with the model
            predicted_noise = model(noisy_images, t)
            loss = mse_loss(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
        
        if epoch % 5 == 0:
            # Save the model checkpoint after 5 epoch.
            torch.save(model.state_dict(), f"ddpm_mnist_epoch{epoch+1}.pth")




