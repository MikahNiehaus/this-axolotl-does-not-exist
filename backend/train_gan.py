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
from models.git_model_handler import GitModelHandler
from tqdm import tqdm
import argparse

print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[INFO] Using CPU")

# --- CONFIG ---
# Use absolute path to avoid path resolution issues
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
IMG_SIZE = 32  # Start small, change this to upscale later
BATCH_SIZE = 32
EPOCHS = 10000
LEARNING_RATE = 2e-4
Z_DIM = 100
CHECKPOINT_PATH = os.path.join(DATA_DIR, 'gan_checkpoint.pth')
SAMPLE_DIR = os.path.join(DATA_DIR, 'gan_samples')
FULL_MODEL_PATH = os.path.join(DATA_DIR, 'gan_full_model.pth')
print(f"[PATHS] DATA_DIR: {DATA_DIR}")
print(f"[PATHS] CHECKPOINT_PATH: {CHECKPOINT_PATH}")
print(f"[PATHS] FULL_MODEL_PATH: {FULL_MODEL_PATH}")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_INTERVAL = 100

os.makedirs(SAMPLE_DIR, exist_ok=True)

# --- DATASET ---
class AxolotlDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.reload_files()
        
    def reload_files(self):
        self.files = glob.glob(os.path.join(self.folder, '*'))
        if len(self.files) == 0:
            raise RuntimeError(f"No files found in {self.folder}")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        # Robust to missing/corrupted files
        try:
            img = Image.open(self.files[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Warning: failed to load {self.files[idx]}: {e}")
            # Remove the bad file from the list
            del self.files[idx]
            # If no files left, raise
            if not self.files:
                raise RuntimeError("No valid images left in dataset!")
            # If too many files are missing, reload file list from disk
            if len(self.files) < 10:
                print("Reloading file list from disk due to too many missing files...")
                self.reload_files()
            # Clamp idx to valid range
            idx = idx % len(self.files)
            return self.__getitem__(idx)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # Use fill=1 to fill with white instead of black during rotation and prevent black artifacts
    transforms.RandomRotation(20, fill=1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    # Limit shear to prevent black edges
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=5, fill=1),
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
        
        # Initialize the Git model handler for automatic pushes
        self.git_push_interval = 1000  # Push to Git every 1000 epochs
        
        # Set up Git integration if possible
        self.git_enabled = self._check_git_available()
        if self.git_enabled:
            try:
                self.git_handler = GitModelHandler(CHECKPOINT_PATH)
                print(f"[GIT] Git integration enabled - Model will be pushed to main branch every {self.git_push_interval} epochs")
            except Exception as e:
                print(f"[GIT] Warning: Could not initialize Git handler: {str(e)}")
                self.git_enabled = False
        else:
            print("[GIT] Git integration disabled - Git not available or not in a Git repository")
        
        self.resolution_history = []  # Track all resolutions and epochs
        print("[SUMMARY] GANTrainer will:")
        print(" - Log the current image resolution at every epoch and checkpoint.")
        print(" - Continuously increase resolution on overfitting.")
        print(" - Increase checkpointing if VRAM runs out.")
        print(" - Save a full model and keep a history of all checkpoint epochs and resolutions.")
        
    def _check_git_available(self):
        """Check if Git is available and we're in a Git repository"""
        try:
            # Try to execute a simple git command
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"], 
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False

    def upscale(self):
        new_size = min(self.img_size * self.upscale_factor, self.max_img_size)
        if new_size == self.img_size:
            print("[INFO] Already at max image size. Continuing with quality improvements.")
            # Refresh the model parameters to break out of local minima without changing resolution
            self.opt_G = optim.Adam(self.G.parameters(), lr=LEARNING_RATE * 0.8, betas=(0.5, 0.999))
            self.opt_D = optim.Adam(self.D.parameters(), lr=LEARNING_RATE * 0.8, betas=(0.5, 0.999))
            # Still return True to reset no_improve counter and continue training
            return True
            
        print(f"[INFO] Upscaling GAN from {self.img_size} to {new_size}.")
        # Recreate Generator and Discriminator with new size
        self.img_size = new_size
        self.G = Generator(z_dim=self.z_dim, img_channels=3, img_size=new_size).to(self.device)
        self.D = Discriminator(img_channels=3, img_size=new_size).to(self.device)
        self.opt_G = optim.Adam(self.G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        self.fixed_noise = torch.randn(16, self.z_dim, 1, 1, device=self.device)
        print(f"[RESOLUTION] New training resolution: {self.img_size}x{self.img_size}")
        return True

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save VRAM memory at the cost of computation speed"""
        try:
            # Import needed for gradient checkpointing
            import torch.utils.checkpoint
            
            # Enable checkpointing on Generator
            self.G.gradient_checkpointing_enable()
            print(f"[VRAM] Enabled gradient checkpointing on Generator (level {self.checkpointing_level})")
            
            # Enable checkpointing on Discriminator  
            self.D.gradient_checkpointing_enable()
            print(f"[VRAM] Enabled gradient checkpointing on Discriminator (level {self.checkpointing_level})")
            
            # Additional memory saving measures based on checkpoint level
            if self.checkpointing_level >= 2:
                # Reduce batch size
                global BATCH_SIZE
                new_batch_size = max(4, BATCH_SIZE // 2)
                if new_batch_size != BATCH_SIZE:
                    print(f"[VRAM] Reducing batch size from {BATCH_SIZE} to {new_batch_size}")
                    BATCH_SIZE = new_batch_size
                    
            if self.checkpointing_level >= 3:
                # Use mixed precision training if available
                try:
                    from torch.cuda.amp import autocast, GradScaler
                    self.use_amp = True
                    self.scaler = GradScaler()
                    print("[VRAM] Enabled mixed precision training")
                except ImportError:
                    print("[VRAM] Mixed precision training not available")
                    
            return True
        except Exception as e:
            print(f"[VRAM] Error enabling gradient checkpointing: {str(e)}")
            return False

    def save_checkpoint(self, epoch):
        """Save model checkpoint and push to Git at specified intervals"""
        # Get clean model states first
        self.G.gradient_checkpointing_disable()
        self.D.gradient_checkpointing_disable()
        
        # Use a temporary checkpoint path to avoid corrupting the main checkpoint if saving fails
        temp_checkpoint_path = CHECKPOINT_PATH + ".tmp"
        try:
            # Save the checkpoint to temporary file first
            torch.save({
                'G': self.G.state_dict(),
                'D': self.D.state_dict(),
                'opt_G': self.opt_G.state_dict(),
                'opt_D': self.opt_D.state_dict(),
                'epoch': epoch,
                'img_size': self.img_size  # Store current image size for proper restoration
            }, temp_checkpoint_path)
            
            # Rename temp file to actual checkpoint path (safer file operation)
            if os.path.exists(temp_checkpoint_path):
                import shutil
                shutil.move(temp_checkpoint_path, CHECKPOINT_PATH)
                
            print(f"[Checkpoint] Saved checkpoint at epoch {epoch+1}")
                
            # Push to Git every git_push_interval epochs
            if self.git_enabled and (epoch + 1) % self.git_push_interval == 0:
                print(f"[GIT] Pushing model checkpoint at epoch {epoch+1} to Git main branch...")
                try:
                    self.git_handler.update_model_in_git(epoch_num=epoch+1)
                except Exception as e:
                    print(f"[GIT] Warning: Failed to push model to Git: {str(e)}")
                    print("[GIT] Continuing training without Git push")
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {str(e)}")
            if os.path.exists(temp_checkpoint_path):
                os.remove(temp_checkpoint_path)
        
        # Log resolution history
        self.resolution_history.append({'epoch': epoch+1, 'img_size': self.img_size})
        print(f"[CHECKPOINT] Epoch {epoch+1} | Resolution: {self.img_size}x{self.img_size}")

    def save_full_model(self, epoch, manual=False):
        """Save the full model (not just checkpoint) and push to Git"""
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Make sure we're using absolute paths
        full_model_abs_path = os.path.abspath(FULL_MODEL_PATH)
        print(f"[FullModel] Saving full model to: {full_model_abs_path}")
        
        # Save the full model (generator, discriminator, optimizer states, etc.)
        torch.save({
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'epoch': epoch,
            'img_size': self.img_size,
            'resolution_history': self.resolution_history
        }, full_model_abs_path)
        
        # Verify the file was created
        if os.path.exists(full_model_abs_path):
            file_size = os.path.getsize(full_model_abs_path)
            print(f"[FullModel] Saved full model at epoch {epoch+1} to {full_model_abs_path} (Size: {file_size} bytes)")
        # Push only the full model to git
        if self.git_enabled:
            try:
                handler = GitModelHandler(FULL_MODEL_PATH)
                comment = f"Full GAN model at epoch {epoch+1}" if not manual else "Manual full GAN model save"
                handler.update_model_in_git(epoch_num=epoch+1)
                print(f"[GIT] Pushed full model to Git with comment: {comment}")
            except Exception as e:
                print(f"[GIT] Warning: Failed to push full model to Git: {str(e)}")

    def load_checkpoint(self):
        if os.path.exists(CHECKPOINT_PATH):
            try:
                checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
                
                # Check if the checkpoint has a different image size 
                saved_img_size = checkpoint.get('img_size', self.img_size)
                if saved_img_size != self.img_size:
                    print(f"[INFO] Checkpoint has image size {saved_img_size}, current is {self.img_size}")
                    print(f"[INFO] Recreating models with saved image size {saved_img_size}")
                    # Recreate models with correct size
                    self.img_size = saved_img_size
                    self.G = Generator(z_dim=self.z_dim, img_channels=3, img_size=saved_img_size).to(self.device)
                    self.D = Discriminator(img_channels=3, img_size=saved_img_size).to(self.device)
                    self.opt_G = optim.Adam(self.G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
                    self.opt_D = optim.Adam(self.D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
                    self.fixed_noise = torch.randn(16, self.z_dim, 1, 1, device=self.device)
                
                # Load state dictionaries
                self.G.load_state_dict(checkpoint['G'])
                self.D.load_state_dict(checkpoint['D'])
                self.opt_G.load_state_dict(checkpoint['opt_G'])
                self.opt_D.load_state_dict(checkpoint['opt_D'])
                
                # Make sure gradient checkpointing is disabled for loaded models initially
                if hasattr(self.G, 'gradient_checkpointing_disable'):
                    self.G.gradient_checkpointing_disable()
                if hasattr(self.D, 'gradient_checkpointing_disable'):
                    self.D.gradient_checkpointing_disable()
                
                self.start_epoch = checkpoint['epoch'] + 1
                print(f"[INFO] Successfully loaded checkpoint from {CHECKPOINT_PATH} at epoch {self.start_epoch}")
            except Exception as e:
                print(f"[WARN] Error loading checkpoint: {str(e)}. Starting from scratch.")
                self.start_epoch = 0
        else:
            print("No checkpoint found, starting fresh.")
            self.start_epoch = 0

    def generate_sample(self, epoch):
        """Generate a sample using VRAM-safe approach"""
        try:
            # Set eval mode and disable checkpointing to ensure clean output
            self.G.eval()
            if hasattr(self.G, 'gradient_checkpointing_disable'):
                self.G.gradient_checkpointing_disable()

            # Generate single image first (smaller memory footprint)
            with torch.no_grad():
                fake = self.G(self.fixed_noise[:1]).detach().cpu()
                save_image(fake, os.path.join(SAMPLE_DIR, f'sample_epoch.png'), normalize=True)

                # Generate a grid of samples and overwrite the same file
                try:
                    torch.cuda.empty_cache()  # Clear memory first
                    fake = self.G(self.fixed_noise).detach().cpu()
                    save_image(fake, os.path.join(SAMPLE_DIR, 'sample_grid.png'), normalize=True, nrow=4)
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print("[VRAM] Grid sample generation skipped due to memory constraints")
                    else:
                        print(f"[ERROR] Grid sample generation error: {str(e)}")
        except Exception as e:
            print(f"[ERROR] Sample generation failed: {str(e)}")
            # Fall back to CPU if necessary
            try:
                print("[VRAM] Attempting to generate sample on CPU...")
                self.G = self.G.cpu()
                with torch.no_grad():
                    fake = self.G(self.fixed_noise[:1].cpu()).detach()
                    save_image(fake, os.path.join(SAMPLE_DIR, 'sample_grid.png'), normalize=True)
                self.G = self.G.to(self.device)
            except Exception as e2:
                print(f"[CRITICAL] CPU fallback failed too: {str(e2)}")
        finally:
            # Restore model to training state
            self.G.train()
            # Re-enable gradient checkpointing if it was active
            if hasattr(self, 'checkpointing_level') and self.checkpointing_level > 0:
                self.enable_gradient_checkpointing()

    def train(self, train_loader, epochs=float('inf'), sample_interval=1000):
        # Set epochs to infinity to train forever
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
                
                # Setup mixed precision if enabled by VRAM optimization
                use_amp = hasattr(self, 'use_amp') and self.use_amp
                
                for real in pbar:
                    real = real.to(self.device)
                    batch_size = real.size(0)
                    noise = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
                    
                    # Train Discriminator with optional mixed precision
                    self.opt_D.zero_grad()
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            fake = self.G(noise)
                            D_real = self.D(real).view(-1)
                            D_fake = self.D(fake.detach()).view(-1)
                            loss_D = self.criterion(D_real, torch.ones_like(D_real)) + \
                                    self.criterion(D_fake, torch.zeros_like(D_fake))
                        self.scaler.scale(loss_D).backward()
                        self.scaler.step(self.opt_D)
                    else:
                        fake = self.G(noise)
                        D_real = self.D(real).view(-1)
                        D_fake = self.D(fake.detach()).view(-1)
                        loss_D = self.criterion(D_real, torch.ones_like(D_real)) + \
                                self.criterion(D_fake, torch.zeros_like(D_fake))
                        loss_D.backward()
                        self.opt_D.step()
                    
                    # Train Generator with optional mixed precision
                    self.opt_G.zero_grad()
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            output = self.D(fake).view(-1)
                            loss_G = self.criterion(output, torch.ones_like(output))
                        self.scaler.scale(loss_G).backward()
                        self.scaler.step(self.opt_G)
                        self.scaler.update()
                    else:
                        output = self.D(fake).view(-1)
                        loss_G = self.criterion(output, torch.ones_like(output))
                        loss_G.backward()
                        self.opt_G.step()
                    
                    D_losses.append(loss_D.item())
                    G_losses.append(loss_G.item())
                    pbar.set_postfix({"D_loss": loss_D.item(), "G_loss": loss_G.item()})
                avg_D_loss = sum(D_losses) / len(D_losses)
                avg_G_loss = sum(G_losses) / len(G_losses)
                print(f"[Epoch {epoch+1}] D_loss: {avg_D_loss:.4f} | G_loss: {avg_G_loss:.4f} | Resolution: {self.img_size}x{self.img_size}")
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
                            # Use fill=1 to fill with white instead of black during rotation and prevent black artifacts
                            transforms.RandomRotation(20, fill=1),
                            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                            # Limit shear to prevent black edges
                            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=5, fill=1),
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
                    print(f"[Epoch {epoch+1}] Saving sample and checkpoint...")
                    self.generate_sample(epoch+1)
                    self.save_checkpoint(epoch)
                    
                    # Also update the manual sample image (now on the same schedule as checkpoints)
                    self.G.eval()
                    with torch.no_grad():
                        fake = self.G(self.fixed_noise[:1]).detach().cpu()
                        save_image(fake, os.path.join(SAMPLE_DIR, 'sample_epochmanual.png'), normalize=True)
                    self.G.train()
                
                # Create full model file at epoch 1 and then every git_push_interval epochs
                if (epoch + 1) == 1 or (epoch + 1) % self.git_push_interval == 0:
                    self.save_full_model(epoch)
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
    parser.add_argument('command', nargs='?', default='train', choices=['train', 'sample', 'save_full_model'], help="'train' to train, 'sample' to generate a sample image only, 'save_full_model' to manually save and push full model")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    trainer = GANTrainer(img_size=IMG_SIZE, z_dim=Z_DIM, lr=LEARNING_RATE, batch_size=BATCH_SIZE, device=DEVICE)

    if args.command == 'sample':
        print("Generating a full image sample using the current generator...")
        trainer.load_checkpoint()
        
        # Make multiple attempts to generate samples with increasing VRAM optimization if needed
        try:
            trainer.generate_sample('manual')
            print(f"Sample saved to {os.path.join(SAMPLE_DIR, 'sample_epochmanual.png')}")
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("[VRAM] CUDA out of memory detected. Attempting with gradient checkpointing...")
                trainer.checkpointing_level = 1
                trainer.enable_gradient_checkpointing()
                try:
                    trainer.generate_sample('manual')
                    print(f"Sample saved to {os.path.join(SAMPLE_DIR, 'sample_epochmanual.png')}")
                except Exception as e2:
                    print(f"[ERROR] Failed to generate sample even with checkpointing: {str(e2)}")
                    print("[VRAM] Falling back to CPU generation...")
                    trainer.G = trainer.G.cpu()
                    trainer.fixed_noise = trainer.fixed_noise.cpu()
                    trainer.generate_sample('manual_cpu')
                    print(f"Sample saved to {os.path.join(SAMPLE_DIR, 'sample_epochmanual_cpu.png')}")
            else:
                print(f"[ERROR] Failed to generate sample: {e}")
    elif args.command == 'save_full_model':
        trainer.load_checkpoint()
        trainer.save_full_model(trainer.start_epoch, manual=True)
    else:
        trainer.train(train_loader, epochs=EPOCHS, sample_interval=SAMPLE_INTERVAL)
