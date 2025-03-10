import torch 
import torch.nn as nn 
from gans.models.gans import Generator, Discriminator, weights_init_normal
import torch.optim as optim
import argparse
from tqdm import tqdm
from utils.datamanger import DataManager
import os 

checkpoint_dir = "gans/checkpoints/vanilla_gans/"
os.makedirs(checkpoint_dir, exist_ok=True)

def train_bce(args):
    """
    Train a vanilla GAN using Binary Cross-Entropy loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_shape = (1, 28, 28)
    train_loader, _ = DataManager.get_mnist_dataloaders(batch_size=args.batch_size)
    
    # Initialize generator and discriminator
    generator = Generator(args.latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape, use_sigmoid=True).to(device)
    
    # Apply custom weight initialization
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
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


def train_wgan(args):
    """
    Train a Wasserstein GAN.
    Uses RMSprop optimizer, multiple discriminator updates (n_critic), and weight clipping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_shape = (1, 28, 28)
    train_loader, _ = DataManager.get_mnist_dataloaders(batch_size=args.batch_size)
    
    # For WGAN, do not use Sigmoid activation at the output of the discriminator
    generator = Generator(args.latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape, use_sigmoid=False).to(device)
    
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    # Setup optimizers for WGAN (RMSprop)
    optimizer_G = optim.RMSprop(generator.parameters(), lr=args.lr)
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=args.lr)
    
    print("Starting WGAN training...")
    for epoch in range(args.n_epochs):
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.n_epochs}") as pbar:
            for i, (imgs, _) in enumerate(train_loader):
                batch_size = imgs.size(0)
                real_imgs = imgs.to(device)
                
                # ---------------------
                # Train Discriminator (Critic) - n_critic times
                # ---------------------
                for _ in range(args.n_critic):
                    optimizer_D.zero_grad()
                    # Sample noise and generate fake images
                    noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
                    fake_imgs = generator(noise).detach()
                    # Wasserstein loss: maximize D(real) - D(fake)
                    d_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
                    d_loss.backward()
                    optimizer_D.step()
                    
                    # Weight clipping to enforce Lipschitz constraint
                    for p in discriminator.parameters():
                        p.data.clamp_(-args.clip_value, args.clip_value)
                
                # -----------------
                # Train Generator
                # -----------------
                optimizer_G.zero_grad()
                noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
                fake_imgs = generator(noise)
                # Generator loss: minimize -D(fake)
                g_loss = -torch.mean(discriminator(fake_imgs))
                g_loss.backward()
                optimizer_G.step()
                
                pbar.set_postfix(G_Loss=g_loss.item(), D_Loss=d_loss.item())
                pbar.update(1)
        
        # Save a checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"wgan_checkpoint_epoch_{epoch+1}.pth")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam: decay of first order momentum")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam: decay of second order momentum")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of the latent space")
    parser.add_argument("--loss_type", type=str, default="bce", choices=["bce", "wgan"],
                        help="Loss type: 'bce' for Binary Cross-Entropy, 'wgan' for Wasserstein GAN")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="Number of training steps for discriminator per generator step (for WGAN)")
    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="Lower and upper clip value for discriminator weights (for WGAN)")
    args = parser.parse_args()
    
    if args.loss_type == "bce":
        train_bce(args)
    elif args.loss_type == "wgan":
        train_wgan(args)