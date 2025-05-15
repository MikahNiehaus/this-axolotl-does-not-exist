import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, features_g=64, img_size=32):
        super().__init__()
        self.img_size = img_size
        layers = []
        # 32x32: 4 layers, 64x64+: 5 layers
        if img_size == 32:
            layers = [
                nn.ConvTranspose2d(z_dim, features_g*4, 4, 1, 0, bias=False), # 1x1 -> 4x4
                nn.BatchNorm2d(features_g*4),
                nn.ReLU(True),
                nn.ConvTranspose2d(features_g*4, features_g*2, 4, 2, 1, bias=False), # 4x4 -> 8x8
                nn.BatchNorm2d(features_g*2),
                nn.ReLU(True),
                nn.ConvTranspose2d(features_g*2, features_g, 4, 2, 1, bias=False), # 8x8 -> 16x16
                nn.BatchNorm2d(features_g),
                nn.ReLU(True),
                nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1, bias=False), # 16x16 -> 32x32
                nn.Tanh()
            ]
        else:
            # Default to 64x64/128x128 DCGAN
            layers = [
                nn.ConvTranspose2d(z_dim, features_g*8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(features_g*8),
                nn.ReLU(True),
                nn.ConvTranspose2d(features_g*8, features_g*4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_g*4),
                nn.ReLU(True),
                nn.ConvTranspose2d(features_g*4, features_g*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_g*2),
                nn.ReLU(True),
                nn.ConvTranspose2d(features_g*2, features_g, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_g),
                nn.ReLU(True),
                nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            ]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features_d=64, img_size=32):
        super().__init__()
        self.img_size = img_size
        layers = []
        # 32x32: 4 layers, 64x64+: 5 layers
        if img_size == 32:
            layers = [
                nn.Conv2d(img_channels, features_d, 4, 2, 1, bias=False), # 32x32 -> 16x16
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(features_d, features_d*2, 4, 2, 1, bias=False), # 16x16 -> 8x8
                nn.BatchNorm2d(features_d*2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(features_d*2, features_d*4, 4, 2, 1, bias=False), # 8x8 -> 4x4
                nn.BatchNorm2d(features_d*4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(features_d*4, 1, 4, 1, 0, bias=False), # 4x4 -> 1x1
                nn.Sigmoid()
            ]
        else:
            layers = [
                nn.Conv2d(img_channels, features_d, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(features_d, features_d*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d*2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(features_d*2, features_d*4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d*4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(features_d*4, features_d*8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features_d*8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(features_d*8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            ]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
