from gans.models.dcgans import Generator, Discriminator, weights_init
import torch 
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm
from utils.datamanger import DataManager
import os 
import argparse

# Directory to store checkpoints
checkpoint_dir = "gans/checkpoints/dc_gans/"
os.makedirs(checkpoint_dir, exist_ok=True)

def train_bce(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_shape = (1, 28, 28)
    train_loader, _ = DataManager.get_cifar10_dataloaders(batch_size=args.batch_size)
    
    # Initialize generator and discriminator
    generator = Generator(args.latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape, use_sigmoid=True).to(device)
    
    # Apply custom weight initialization
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss function and optimizers for BCE training
    adversarial_loss = nn.BCELoss().to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    
    print("Starting BCE training...")
    for epoch in range(args.n_epochs):
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.n_epochs}") as pbar:
            for i, (imgs, _) in enumerate(train_loader):
                batch_size = imgs.size(0)
                real_imgs = imgs.to(device)
                
                # Create labels for real and fake images
                real_labels = torch.ones(batch_size, 1, device=device, dtype=torch.float)
                fake_labels = torch.zeros(batch_size, 1, device=device, dtype=torch.float)
                
                # -----------------
                # Train Generator
                # -----------------
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, args.latent_dim, device=device)
                gen_imgs = generator(z)
                g_loss = adversarial_loss(discriminator(gen_imgs), real_labels)
                g_loss.backward()
                optimizer_G.step()
                
                # ------------------
                # Train Discriminator
                # ------------------
                optimizer_D.zero_grad()
                real_loss = adversarial_loss(discriminator(real_imgs), real_labels)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake_labels)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
                
                pbar.set_postfix(G_Loss=g_loss.item(), D_Loss=d_loss.item())
                pbar.update(1)
        
        # Save a checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"bce_checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'g_loss': g_loss.item(),
                'd_loss': d_loss.item()
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam: decay of first order momentum")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam: decay of second order momentum")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of the latent space")
    args = parser.parse_args()
    
    train_bce(args)
