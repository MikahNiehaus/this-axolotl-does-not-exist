import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, features_g=64, img_size=32):
        super().__init__()
        self.img_size = img_size
        self.z_dim = z_dim
        self.features_g = features_g
        self.img_channels = img_channels
        self.use_checkpointing = False
        
        # Build network
        self._build_network()
        
    def _build_network(self):
        # 32x32: 4 layers, 64x64+: 5 layers
        layers = []
        if self.img_size == 32:
            self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(self.z_dim, self.features_g*4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(self.features_g*4),
                nn.ReLU(True)
            )
            self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(self.features_g*4, self.features_g*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.features_g*2),
                nn.ReLU(True)
            )
            self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(self.features_g*2, self.features_g, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.features_g),
                nn.ReLU(True)
            )
            self.layer4 = nn.Sequential(
                nn.ConvTranspose2d(self.features_g, self.img_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )
            # Also create a combined network for non-checkpointing mode
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        else:
            # Default to 64x64/128x128 DCGAN
            self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(self.z_dim, self.features_g*8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(self.features_g*8),
                nn.ReLU(True)
            )
            self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(self.features_g*8, self.features_g*4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.features_g*4),
                nn.ReLU(True)
            )
            self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(self.features_g*4, self.features_g*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.features_g*2),
                nn.ReLU(True)
            )
            self.layer4 = nn.Sequential(
                nn.ConvTranspose2d(self.features_g*2, self.features_g, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.features_g),
                nn.ReLU(True)
            )
            self.layer5 = nn.Sequential(
                nn.ConvTranspose2d(self.features_g, self.img_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )
            # Also create a combined network for non-checkpointing mode
            layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
            
        # Create a single sequential module for faster inference when not using checkpointing
        self.net = nn.Sequential(*[m for layer in layers for m in layer])
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to save VRAM"""
        self.use_checkpointing = True
        return self
        
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.use_checkpointing = False
        return self
    
    def _checkpoint_forward(self, x):
        """Forward pass with gradient checkpointing for memory efficiency"""
        # Use torch.utils.checkpoint to save memory during training
        # Each layer is checkpointed separately
        if self.img_size == 32:
            x = torch.utils.checkpoint.checkpoint(lambda x: self.layer1(x), x)
            x = torch.utils.checkpoint.checkpoint(lambda x: self.layer2(x), x)
            x = torch.utils.checkpoint.checkpoint(lambda x: self.layer3(x), x)
            x = self.layer4(x)  # No need to checkpoint the final layer
        else:
            x = torch.utils.checkpoint.checkpoint(lambda x: self.layer1(x), x)
            x = torch.utils.checkpoint.checkpoint(lambda x: self.layer2(x), x)
            x = torch.utils.checkpoint.checkpoint(lambda x: self.layer3(x), x)
            x = torch.utils.checkpoint.checkpoint(lambda x: self.layer4(x), x)
            x = self.layer5(x)  # No need to checkpoint the final layer
        return x
    
    def forward(self, x):
        if self.use_checkpointing and self.training:
            return self._checkpoint_forward(x)
        else:
            return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features_d=64, img_size=32):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.features_d = features_d
        self.use_checkpointing = False
        
        # Build network
        self._build_network()
        
    def _build_network(self):
        # 32x32: 4 layers, 64x64+: 5 layers
        layers = []
        if self.img_size == 32:
            self.layer1 = nn.Sequential(
                nn.Conv2d(self.img_channels, self.features_d, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(self.features_d, self.features_d*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.features_d*2),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(self.features_d*2, self.features_d*4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.features_d*4),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(self.features_d*4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
            # Also create a combined network for non-checkpointing mode
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(self.img_channels, self.features_d, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(self.features_d, self.features_d*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.features_d*2),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(self.features_d*2, self.features_d*4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.features_d*4),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(self.features_d*4, self.features_d*8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.features_d*8),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layer5 = nn.Sequential(
                nn.Conv2d(self.features_d*8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
            # Also create a combined network for non-checkpointing mode
            layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
            
        # Create a single sequential module for faster inference when not using checkpointing
        self.net = nn.Sequential(*[m for layer in layers for m in layer])
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to save VRAM"""
        self.use_checkpointing = True
        return self
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.use_checkpointing = False
        return self
    
    def _checkpoint_forward(self, x):
        """Forward pass with gradient checkpointing for memory efficiency"""
        # Use torch.utils.checkpoint to save memory during training
        if self.img_size == 32:
            x = torch.utils.checkpoint.checkpoint(lambda x: self.layer1(x), x)
            x = torch.utils.checkpoint.checkpoint(lambda x: self.layer2(x), x)
            x = torch.utils.checkpoint.checkpoint(lambda x: self.layer3(x), x)
            x = self.layer4(x)  # No need to checkpoint the final layer
        else:
            x = torch.utils.checkpoint.checkpoint(lambda x: self.layer1(x), x)
            x = torch.utils.checkpoint.checkpoint(lambda x: self.layer2(x), x)
            x = torch.utils.checkpoint.checkpoint(lambda x: self.layer3(x), x)
            x = torch.utils.checkpoint.checkpoint(lambda x: self.layer4(x), x)
            x = self.layer5(x)  # No need to checkpoint the final layer
        return x
    
    def forward(self, x):
        if self.use_checkpointing and self.training:
            return self._checkpoint_forward(x)
        else:
            return self.net(x)
