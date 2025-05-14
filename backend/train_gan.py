import os
import glob
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

# --- CONFIG ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10000  # Will stop early if overfitting
PATIENCE = 10  # Early stopping patience
LEARNING_RATE = 2e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DATASET ---
class AxolotlDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.files = glob.glob(os.path.join(folder, '*'))
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_ds = AxolotlDataset(TRAIN_DIR, transform)
test_ds = AxolotlDataset(TEST_DIR, transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# --- SIMPLE GENERATOR/DISCRIMINATOR (DCGAN) ---
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, features_g=64):
        super().__init__()
        self.net = nn.Sequential(
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
        )
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features_d=64):
        super().__init__()
        self.net = nn.Sequential(
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
        )
    def forward(self, x):
        return self.net(x)

# --- TRAINING LOOP WITH EARLY STOPPING ---
def train():
    z_dim = 100
    G = Generator(z_dim).to(DEVICE)
    D = Discriminator().to(DEVICE)
    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    best_test_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        G.train()
        D.train()
        for real in train_loader:
            real = real.to(DEVICE)
            batch_size = real.size(0)
            noise = torch.randn(batch_size, z_dim, 1, 1, device=DEVICE)
            fake = G(noise)
            # Train Discriminator
            D_real = D(real).view(-1)
            D_fake = D(fake.detach()).view(-1)
            loss_D = criterion(D_real, torch.ones_like(D_real)) + \
                     criterion(D_fake, torch.zeros_like(D_fake))
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            # Train Generator
            output = D(fake).view(-1)
            loss_G = criterion(output, torch.ones_like(output))
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
        # Validation (overfitting detection)
        G.eval()
        D.eval()
        test_loss = 0
        with torch.no_grad():
            for real in test_loader:
                real = real.to(DEVICE)
                batch_size = real.size(0)
                noise = torch.randn(batch_size, z_dim, 1, 1, device=DEVICE)
                fake = G(noise)
                D_real = D(real).view(-1)
                D_fake = D(fake).view(-1)
                loss_D = criterion(D_real, torch.ones_like(D_real)) + \
                         criterion(D_fake, torch.zeros_like(D_fake))
                test_loss += loss_D.item()
        test_loss /= len(test_loader)
        print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}")
        # Early stopping if overfitting
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(G.state_dict(), os.path.join(DATA_DIR, 'best_generator.pth'))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping: Overfitting detected.")
                break
        # Save sample images
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                fake = G(torch.randn(16, z_dim, 1, 1, device=DEVICE))
                save_image(fake, os.path.join(DATA_DIR, f'fake_samples_epoch{epoch+1}.png'), normalize=True)
    print("Training complete.")

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    train()
