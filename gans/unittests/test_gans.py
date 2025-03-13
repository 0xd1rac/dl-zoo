
import unittest
from model_archs.gans import Generator, Discriminator
import torch 

class TestGANs(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 128
        self.img_shape = (1, 28, 28)
        self.batch_size = 10
        self.g = Generator(latent_dim=self.latent_dim, img_shape=self.img_shape)
        self.d = Discriminator(img_shape=self.img_shape)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.g.to(self.device)
        self.d.to(self.device)

        self.dummy_img = torch.rand(self.batch_size, *self.img_shape, device=self.device)
    
    def test_generator_forward_shape(self):
        """Test that Generator forward pass generates an image"""
        target_shape = torch.Size((self.batch_size, *self.img_shape))  
        z = torch.rand(self.batch_size, self.latent_dim, device=self.device)
        gen_imgs = self.g(z)
        self.assertEqual(gen_imgs.shape, target_shape,
                         msg="Generator output shape is wrong!!")        

    def test_discriminator_forward_shape(self):
        """Test that discriminator forward pass generates a prediction"""
        preds = self.d(self.dummy_img)
        
        # Expected output shape: [batch_size, 1]
        target_shape = torch.Size([self.batch_size, 1])

        self.assertEqual(preds.shape, target_shape, "Discriminator output shape is wrong!!")

        # Check if the output is in the valid range (assuming a sigmoid output)
        self.assertTrue(torch.all((preds >= 0) & (preds <= 1)), "Discriminator output is out of range!!")

if __name__ == "__main__":
    unittest.main()