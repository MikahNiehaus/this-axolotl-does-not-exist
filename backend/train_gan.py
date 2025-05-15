import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from models.gan_modules import Generator, Discriminator
from tqdm import tqdm
import argparse

# --- CONFIG ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
IMG_SIZE = 32  # Start small, change this to upscale later
BATCH_SIZE = 32
EPOCHS = 10000
LEARNING_RATE = 2e-4
Z_DIM = 100
CHECKPOINT_PATH = os.path.join(DATA_DIR, 'gan_checkpoint.pth')
SAMPLE_DIR = os.path.join(DATA_DIR, 'gan_samples')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_INTERVAL = 1000

os.makedirs(SAMPLE_DIR, exist_ok=True)

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
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_ds = AxolotlDataset(TRAIN_DIR, transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# --- GAN TRAINER CLASS ---
class GANTrainer:
    def __init__(self, img_size=32, z_dim=100, lr=2e-4, batch_size=32, device='cpu'):
        self.img_size = img_size
        self.z_dim = z_dim
        self.device = device
        self.G = Generator(z_dim=z_dim, img_channels=3, img_size=img_size).to(device)
        self.D = Discriminator(img_channels=3, img_size=img_size).to(device)
        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        self.start_epoch = 0
        self.fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)
        self.overfit_counter = 0
        self.overfit_patience = 10  # Number of epochs to tolerate no improvement before upscaling
        self.last_D_losses = []
        self.last_G_losses = []
        self.upscale_factor = 2  # Double the image size on each upscale
        self.max_img_size = 128  # Maximum image size for upscaling
        self.checkpointing_level = 0  # For VRAM fallback

    def upscale(self):
        new_size = min(self.img_size * self.upscale_factor, self.max_img_size)
        if new_size == self.img_size:
            print("[INFO] Already at max image size, cannot upscale further.")
            return False
        print(f"[INFO] Upscaling GAN from {self.img_size} to {new_size}.")
        # Recreate Generator and Discriminator with new size
        self.img_size = new_size
        self.G = Generator(z_dim=self.z_dim, img_channels=3, img_size=new_size).to(self.device)
        self.D = Discriminator(img_channels=3, img_size=new_size).to(self.device)
        self.opt_G = optim.Adam(self.G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        self.fixed_noise = torch.randn(16, self.z_dim, 1, 1, device=self.device)
        return True

    def enable_gradient_checkpointing(self):
        # Try to enable gradient checkpointing if supported
        if hasattr(self.G, 'gradient_checkpointing_enable'):
            self.G.gradient_checkpointing_enable()
            print(f"[VRAM] Enabled gradient checkpointing on Generator (level {self.checkpointing_level})")
        if hasattr(self.D, 'gradient_checkpointing_enable'):
            self.D.gradient_checkpointing_enable()
            print(f"[VRAM] Enabled gradient checkpointing on Discriminator (level {self.checkpointing_level})")

    def save_checkpoint(self, epoch):
        torch.save({
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'epoch': epoch
        }, CHECKPOINT_PATH)

    def load_checkpoint(self):
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
            self.G.load_state_dict(checkpoint['G'])
            self.D.load_state_dict(checkpoint['D'])
            self.opt_G.load_state_dict(checkpoint['opt_G'])
            self.opt_D.load_state_dict(checkpoint['opt_D'])
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from checkpoint at epoch {self.start_epoch}")
        else:
            print("No checkpoint found, starting fresh.")

    def generate_sample(self, epoch):
        self.G.eval()
        with torch.no_grad():
            fake = self.G(self.fixed_noise[:1]).detach().cpu()  # Only generate one image
            save_image(fake, os.path.join(SAMPLE_DIR, f'sample_epoch{epoch}.png'), normalize=True)
        self.G.train()

    def train(self, train_loader, epochs=10000, sample_interval=1000):
        local_train_loader = train_loader
        self.load_checkpoint()
        epoch = self.start_epoch
        best_val = float('inf')
        no_improve = 0
        prev_G_loss = None
        vram_retry = 0
        while True:
            try:
                D_losses = []
                G_losses = []
                pbar = tqdm(local_train_loader, desc=f"Epoch {epoch+1}", leave=False)
                for real in pbar:
                    real = real.to(self.device)
                    batch_size = real.size(0)
                    noise = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
                    fake = self.G(noise)
                    D_real = self.D(real).view(-1)
                    D_fake = self.D(fake.detach()).view(-1)
                    loss_D = self.criterion(D_real, torch.ones_like(D_real)) + \
                             self.criterion(D_fake, torch.zeros_like(D_fake))
                    self.opt_D.zero_grad()
                    loss_D.backward()
                    self.opt_D.step()
                    output = self.D(fake).view(-1)
                    loss_G = self.criterion(output, torch.ones_like(output))
                    self.opt_G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()
                    D_losses.append(loss_D.item())
                    G_losses.append(loss_G.item())
                    pbar.set_postfix({"D_loss": loss_D.item(), "G_loss": loss_G.item()})
                avg_D_loss = sum(D_losses) / len(D_losses)
                avg_G_loss = sum(G_losses) / len(G_losses)
                print(f"[Epoch {epoch+1}] D_loss: {avg_D_loss:.4f} | G_loss: {avg_G_loss:.4f}")
                # Overfitting/no-improvement heuristic
                if prev_G_loss is not None and avg_G_loss >= prev_G_loss:
                    no_improve += 1
                    print(f"[No improvement] {no_improve}/10 epochs.")
                else:
                    no_improve = 0
                prev_G_loss = avg_G_loss
                if no_improve >= self.overfit_patience:
                    print(f"[INFO] No improvement for {self.overfit_patience} epochs. Attempting to upscale.")
                    if self.upscale():
                        # Update transform and dataloader for new image size
                        global transform, train_ds
                        transform = transforms.Compose([
                            transforms.Resize((self.img_size, self.img_size)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomRotation(20),
                            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5]*3, [0.5]*3)
                        ])
                        train_ds = AxolotlDataset(TRAIN_DIR, transform)
                        local_train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
                        no_improve = 0
                        continue  # Restart training at new resolution
                    else:
                        print("[INFO] Max image size reached. Continuing training at current resolution.")
                        no_improve = 0
                if (epoch + 1) % sample_interval == 0 or epoch == 0:
                    print(f"[Epoch {epoch+1}] Saving sample...")
                    self.generate_sample(epoch+1)
                    self.save_checkpoint(epoch)
                # Update sample_epochmanual.png every 100 epochs
                if (epoch + 1) % 100 == 0:
                    self.G.eval()
                    with torch.no_grad():
                        fake = self.G(self.fixed_noise[:1]).detach().cpu()
                        save_image(fake, os.path.join(SAMPLE_DIR, 'sample_epochmanual.png'), normalize=True)
                    self.G.train()
                epoch += 1
                vram_retry = 0  # Reset VRAM retry counter after successful epoch
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"[VRAM] CUDA out of memory detected. Attempting VRAM fallback (attempt {vram_retry+1}/10)...")
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    self.checkpointing_level += 1
                    self.enable_gradient_checkpointing()
                    print(f"[VRAM] Aggressiveness level: {self.checkpointing_level}")
                    vram_retry += 1
                    if vram_retry > 10:
                        print("[VRAM] Too many VRAM fallback attempts. Exiting training.")
                        raise
                    continue  # Retry epoch with more aggressive checkpointing
                else:
                    raise
        print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN or generate a sample image.')
    parser.add_argument('command', nargs='?', default='train', choices=['train', 'sample'], help="'train' to train, 'sample' to generate a sample image only")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    trainer = GANTrainer(img_size=IMG_SIZE, z_dim=Z_DIM, lr=LEARNING_RATE, batch_size=BATCH_SIZE, device=DEVICE)

    if args.command == 'sample':
        print("Generating a full image sample using the current generator...")
        trainer.load_checkpoint()
        trainer.generate_sample('manual')
        print(f"Sample saved to {os.path.join(SAMPLE_DIR, 'sample_epochmanual.png')}")
    else:
        trainer.train(train_loader, epochs=EPOCHS, sample_interval=SAMPLE_INTERVAL)
