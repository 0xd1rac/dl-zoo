from model_archs.ddpm import Unet, TimeEmbedding, LinearNoiseScheduler
import unittest
import torch 
from torch import Tensor

class TestUnetAndComponents(unittest.TestCase):
    def setUp(self):
        self.model_config = {
            "im_channels":1,
            "down_channels": [32, 64, 128, 128],
            "mid_channels": [128, 128, 128],
            "time_emb_dim": 64,
            "down_sample": [True, True, False],
            "num_down_layers": 2,
            "num_mid_layers": 1,
            "num_up_layers": 2
        }

        self.model = Unet(self.model_config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.batch_size = 4 

        # create dummy input imgs [B, C, H, W]
        self.x = torch.randn(self.batch_size, self.model_config["im_channels"], 28, 28, device=self.device)
        
        # Create dummy time steps as integers (e.g, values between 0 and 999) - one image, one time value, if there are 4 images in the batch, there will be 4 time values
        self.t = torch.randint(0, 1000, (self.batch_size,), device=self.device)

    
    def test_unet_forward_shape(self):
        """Test that the U-Net forward pass returns an output with the same shape as the input."""
        output = self.model(self.x , self.t)
        self.assertEqual(output.shape, self.x.shape, 
                         msg="U-Net output shape should match the input shape."
                         )
        
    def test_time_embedding_shape(self):
        """Test that the TimeEmbedding module returns the correct shape."""
        time_embedder = TimeEmbedding(64)
        t_emb = time_embedder(self.t.to(torch.float32))
        # Expected shape: (batch_size, 64)
        self.assertEqual(t_emb.shape, (self.batch_size, 64),
                         msg="Time embedding output shape must be (batch_size, temb_dim)."
                         )

    def test_linear_noise_scheduler_add_noise(self):
        """Test the add_noise method of the LinearNoiseScheduler."""
        scheduler = LinearNoiseScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        
        # create a t tensor with a fixed time step for all examples 
        t_fixed = torch.full((self.batch_size, ), 500, dtype=torch.long, device=self.device)

        noise = torch.randn_like(self.x)
        noisy_images = scheduler.add_noise(self.x, noise, t_fixed)
        self.assertEqual(noisy_images.shape, self.x.shape,
                         msg="Noisy images should have the same shape as the original images.")

    def test_linear_noise_scheduler_sample_prev_timestep(self):
        """Test the sample_prev_timestep method of the LinearNoiseScheduler"""
        scheduler = LinearNoiseScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        t = 500
        x_t = torch.randn_like(self.x)
        noise_pred = torch.randn_like(self.x)
        x_prev, x_0 = scheduler.sample_prev_timestep(x_t, noise_pred, t)
        self.assertEqual(x_prev.shape, x_t.shape, 
                         msg="Output x_prev should have the same shape as x_t.")
        
        self.assertEqual(x_0.shape, x_t.shape,
                         msg="Output x_0 should have the same shape as x_t.")


if __name__ == "__main__":
    unittest.main()