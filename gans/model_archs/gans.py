import torch 
import torch.nn as nn 
import numpy as np 

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.Linear(256,512),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape, use_sigmoid=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )
        # For BCE-GAN we want the output between 0 and 1
        if self.use_sigmoid:
            self.model.add_module("sigmoid", nn.Sigmoid())

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity.view(-1)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
