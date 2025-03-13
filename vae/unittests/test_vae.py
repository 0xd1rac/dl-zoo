import unittest
from model_archs.vae import Encoder, Decoder, VAE, vae_loss
import torch 

class TestVAE(unittest.TestCase):
    def setUp(self):
        self.input_channels, self.output_chanenls = 3, 3
        self.latent_dim = 128
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.e = Encoder(self.input_channels, self.latent_dim)
        self.d = Decoder(self.latent_dim, self.output_chanenls)
        self.vae = VAE(self.input_channels, self.latent_dim)
        self.e.to(self.device)
        self.d.to(self.device)
        self.batch_size = 4 
        self.dummy_img = torch.randn(self.batch_size, self.input_channels, 64, 64, device=self.device)


    def test_encoder_forward(self):
        """Test that encoder forward pass generates a latent vector"""
        mu, log_var = self.e(self.dummy_img)
        expected_shape = (self.batch_size, self.latent_dim)
        self.assertEqual(mu.shape, expected_shape, 
                         msg=f"Expected mu shape {expected_shape}, but got {mu.shape}")
        self.assertEqual(log_var.shape, expected_shape, 
                         msg=f"Expected log_var shape {expected_shape}, but got {log_var.shape}")

        # Ensure log_var values are valid (not NaN or Inf)
        self.assertTrue(torch.all(torch.isfinite(mu)), "Mu contains NaN or Inf!")
        self.assertTrue(torch.all(torch.isfinite(log_var)), "Log_var contains NaN or Inf!")

    def test_decoder_forward(self):
        mu, log_var = self.e(self.dummy_img)
        z = self.vae.reparameterize(mu, log_var)
        img_recon = self.d(z)
        self.assertEqual(img_recon.shape, self.dummy_img.shape,
                         msg=f"Expected img_recon shape {self.dummy_img.shape}, but got {img_recon.shape}"
                         )

if __name__ == "__main__":
    unittest.main()