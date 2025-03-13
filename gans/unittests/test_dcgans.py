import unittest
from model_archs.dcgans import Generator, Discriminator
import torch 

class TestDCGANs(unittest.TestCase):
    def setUp(self):
        self.img_channels = 3
        self.latent_dim = 128
        self.batch_size = 10
        self.g_feature_map_size = 64
        self.d_feature_map_size = 64
        self.g = Generator(image_channels=self.img_channels, latent_dim=self.latent_dim, generator_feature_map_size=self.g_feature_map_size)
        self.d = Discriminator(image_channels=self.img_channels, discriminator_feature_map_size=self.d_feature_map_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.g.to(self.device)
        self.d.to(self.device)

        self.dummy_img = torch.rand(self.batch_size, self.img_channels, 64, 64, device=self.device)
    
    def test_generator_forward_shape(self):
        """Test that Generator forward pass generates an image"""
        z = torch.rand(self.batch_size, self.latent_dim, 1, 1, device=self.device)
        gen_imgs = self.g(z)
        self.assertEqual(gen_imgs.shape, self.dummy_img.shape,
                         msg="Generator output shape is wrong!!")        

    def test_discriminator_forward_shape(self):
        """Test that discriminator forward pass generates a prediction"""
        preds = self.d(self.dummy_img)
        
        # Expected output shape: [batch_size]
        target_shape = torch.Size([self.batch_size])

        self.assertEqual(preds.shape, target_shape, "Discriminator output shape is wrong!!")

        # Check if the output is in the valid range (assuming a sigmoid output)
        self.assertTrue(torch.all((preds >= 0) & (preds <= 1)), "Discriminator output is out of range!!")

if __name__ == "__main__":
    unittest.main()