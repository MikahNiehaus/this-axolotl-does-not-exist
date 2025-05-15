import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import glob
import random
import gc  # For garbage collection
import time  # For timestamping
import argparse
import traceback  # For detailed error reporting

# Enable CUDA error debugging (for misaligned address and memory issues)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# --- MEMORY MANAGEMENT UTILITIES ---
def print_gpu_memory_status():
    """Print current GPU memory usage for debugging"""
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        used_mem = total_mem - free_mem
        print(f"GPU Memory: {used_mem/(1024**2):.1f}MB used / {total_mem/(1024**2):.1f}MB total")
        return used_mem, total_mem
    else:
        print("GPU not available")
        return 0, 0

def clean_memory():
    """Clean up memory by releasing Python objects and clearing CUDA cache"""
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return True
    return False

# --- CONFIG ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
IMG_SIZE = 64  # Base image size, can be adjusted
BATCH_SIZE = 32
# EPOCHS removed: training is now infinite, controlled by patience only
LEARNING_RATE = 2e-4
RESOLUTION_SCALE = 4.0  # Max res of full image (user can change)
GRADIENT_RES_SCALE = 0.5  # Resolution used for model/training (user can change)
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

# --- PROGRESSIVE SCALING UNET FOR DIFFUSION ---
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=32, resolution_scale=1.0):
        """
        Progressive scaling UNet for diffusion models
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: Base feature multiplier
            resolution_scale: Resolution scaling factor (1.0 = original size, 2.0 = double resolution)
        """
        super().__init__()
        
        self.resolution_scale = resolution_scale
        
        # Adjust features based on resolution for better scaling
        adjusted_features = int(features * min(1.0, 1.0/resolution_scale))
        
        # Store configuration for future scaling
        self.config = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'base_features': features,
            'resolution_scale': resolution_scale
        }
        
        # Memory-efficient encoder with batch norm
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, adjusted_features, 3, 1, 1),
            nn.BatchNorm2d(adjusted_features),
            nn.ReLU(inplace=True)  # inplace operations save memory
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(adjusted_features, adjusted_features*2, 3, 2, 1),
            nn.BatchNorm2d(adjusted_features*2),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(adjusted_features*2, adjusted_features*4, 3, 2, 1),
            nn.BatchNorm2d(adjusted_features*4),
            nn.ReLU(inplace=True)
        )
        
        # Additional encoder layer for higher resolutions
        if resolution_scale >= 1.5:
            self.encoder4 = nn.Sequential(
                nn.Conv2d(adjusted_features*4, adjusted_features*8, 3, 2, 1),
                nn.BatchNorm2d(adjusted_features*8),
                nn.ReLU(inplace=True)
            )
        
        # Memory-efficient middle block with attention for higher resolutions
        if resolution_scale >= 1.5:
            middle_features = adjusted_features*8
            self.middle = nn.Sequential(
                nn.Conv2d(middle_features, middle_features, 3, 1, 1),
                nn.BatchNorm2d(middle_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_features, middle_features, 3, 1, 1),
                nn.BatchNorm2d(middle_features),
                nn.ReLU(inplace=True)
            )
        else:
            middle_features = adjusted_features*4
            self.middle = nn.Sequential(
                nn.Conv2d(middle_features, middle_features, 3, 1, 1),
                nn.BatchNorm2d(middle_features),
                nn.ReLU(inplace=True)
            )
        
        # Memory-efficient decoder with skip connections
        if resolution_scale >= 1.5:
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(middle_features, adjusted_features*4, 4, 2, 1),
                nn.BatchNorm2d(adjusted_features*4),
                nn.ReLU(inplace=True)
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(adjusted_features*4, adjusted_features*2, 4, 2, 1),
                nn.BatchNorm2d(adjusted_features*2),
                nn.ReLU(inplace=True)
            )
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(adjusted_features*2, adjusted_features, 4, 2, 1),
                nn.BatchNorm2d(adjusted_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(middle_features, adjusted_features*2, 4, 2, 1),
                nn.BatchNorm2d(adjusted_features*2),
                nn.ReLU(inplace=True)
            )
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(adjusted_features*2, adjusted_features, 4, 2, 1),
                nn.BatchNorm2d(adjusted_features),
                nn.ReLU(inplace=True)
            )
        
        # Final layer
        self.final = nn.Conv2d(adjusted_features, out_channels, 3, 1, 1)
    
    def forward(self, x):
        # Store input shape for potential upscaling later
        input_shape = x.shape
        
        # Encoder path with skip connections
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        
        # Additional encoding for higher resolutions
        if hasattr(self, 'encoder4'):
            x4 = self.encoder4(x3)
            # Middle block
            x_middle = self.middle(x4)
            # Decoder with skip connections for higher resolution
            x = self.decoder1(x_middle)
            x = self.decoder2(x)
            x = self.decoder3(x)
        else:
            # Middle block for standard resolution
            x_middle = self.middle(x3)
            # Decoder with skip connections for standard resolution
            x = self.decoder1(x_middle)
            x = self.decoder2(x)
        
        # Final output layer
        x = self.final(x)
        
        return x

    def scale_model_resolution(self, new_scale):
        """
        Scale the model to a new resolution by creating a new model and transferring weights
        
        Args:
            new_scale: New resolution scale factor
            
        Returns:
            A new model with scaled resolution and transferred weights where possible
        """
        if abs(new_scale - self.resolution_scale) < 0.001:
            return self
            
        print(f"Scaling model from resolution scale {self.resolution_scale} to {new_scale}")
        
        # Create new model with new resolution scale
        new_model = SimpleUNet(
            in_channels=self.config['in_channels'],
            out_channels=self.config['out_channels'],
            features=self.config['base_features'],
            resolution_scale=new_scale
        )
        
        # Force both models to CPU for safer weight transfer to prevent CUDA memory issues
        original_device = next(self.parameters()).device
        
        # Ensure we're not holding any unnecessary CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Move both models to CPU for safe weight transfer
        self.to('cpu')
        new_model.to('cpu')
        
        # Additional cleanup to ensure no residual GPU memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        print("Transferring weights on CPU to avoid memory alignment issues")
        
        try:
            # Transfer common weights - encoder layers will always exist
            transfer_sequential_weights(self.encoder1, new_model.encoder1)
            transfer_sequential_weights(self.encoder2, new_model.encoder2)
            transfer_sequential_weights(self.encoder3, new_model.encoder3)
            
            # Transfer encoder4 if both have it
            if hasattr(self, 'encoder4') and hasattr(new_model, 'encoder4'):
                transfer_sequential_weights(self.encoder4, new_model.encoder4)
            
            # Transfer middle weights - structures may be different
            try:
                transfer_sequential_weights(self.middle, new_model.middle)
            except Exception as e:
                print(f"Could not transfer middle weights: {e}")
            
            # Transfer decoder weights - may have different structures
            try:
                transfer_sequential_weights(self.decoder1, new_model.decoder1)
                transfer_sequential_weights(self.decoder2, new_model.decoder2)
            except Exception as e:
                print(f"Could not transfer some decoder weights: {e}")
                
            # Transfer decoder3 if both have it
            if hasattr(self, 'decoder3') and hasattr(new_model, 'decoder3'):
                transfer_sequential_weights(self.decoder3, new_model.decoder3)
                
            # Transfer final layer weights if same dimensions
            try:
                transfer_layer_weights(self.final, new_model.final)
            except Exception as e:
                print(f"Could not transfer final layer weights: {e}")
            
            print("Weight transfer completed successfully")
            
            # Return new model to original device if it was on CUDA
            if original_device.type != 'cpu':
                new_model = new_model.to(original_device)
                self.to(original_device)  # Return original model to its device too
                
            return new_model
        
        except Exception as e:
            print(f"Error during model scaling: {e}")
            
            # Return original model to its original device
            if original_device.type != 'cpu':
                self.to(original_device)
                
            # If we fail, create a fresh model without weight transfer
            print("Weight transfer failed, creating new model with fresh weights")
            fresh_model = SimpleUNet(
                in_channels=self.config['in_channels'],
                out_channels=self.config['out_channels'],
                features=self.config['base_features'],
                resolution_scale=new_scale
            )
            
            # Return fresh model to original device if needed
            if original_device.type != 'cpu':
                fresh_model = fresh_model.to(original_device)
            
            return fresh_model
        
def transfer_sequential_weights(source_seq, target_seq):
    """Helper function to transfer weights between sequential layers"""
    for i, (source_layer, target_layer) in enumerate(zip(source_seq, target_seq)):
        if isinstance(source_layer, nn.Conv2d) and isinstance(target_layer, nn.Conv2d):
            try:
                # Transfer conv weights where input/output channels match
                if source_layer.weight.shape == target_layer.weight.shape:
                    target_layer.weight.data.copy_(source_layer.weight.data)
                    if source_layer.bias is not None and target_layer.bias is not None:
                        target_layer.bias.data.copy_(source_layer.bias.data)
            except Exception:
                pass
        elif isinstance(source_layer, nn.BatchNorm2d) and isinstance(target_layer, nn.BatchNorm2d):
            try:
                # Transfer batchnorm weights
                if source_layer.weight.shape == target_layer.weight.shape:
                    target_layer.weight.data.copy_(source_layer.weight.data)
                    target_layer.bias.data.copy_(source_layer.bias.data)
                    target_layer.running_mean.data.copy_(source_layer.running_mean.data)
                    target_layer.running_var.data.copy_(source_layer.running_var.data)
            except Exception:
                pass
                
def transfer_layer_weights(source_layer, target_layer):
    """Helper function to transfer weights between individual layers"""
    try:
        if source_layer.weight.shape == target_layer.weight.shape:
            target_layer.weight.data.copy_(source_layer.weight.data)
            if hasattr(source_layer, 'bias') and hasattr(target_layer, 'bias'):
                if source_layer.bias is not None and target_layer.bias is not None:
                    target_layer.bias.data.copy_(source_layer.bias.data)
    except Exception:
        pass

# --- DIFFUSION UTILS ---
def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

def forward_diffusion_sample(x_0, t, betas):
    # Ensure noise has exactly the same shape as x_0
    noise = torch.randn(x_0.shape, device=x_0.device)
    sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(1 - betas, dim=0))
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - torch.cumprod(1 - betas, dim=0))
    
    # Generate noisy image
    noisy_x = sqrt_alphas_cumprod[t][:, None, None, None] * x_0 + \
              sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise
              
    return noisy_x, noise

# --- TRAINING LOOP WITH PROGRESSIVE SCALING AND CHECKPOINTING ---
def train(resolution_scale=GRADIENT_RES_SCALE, continue_training=True):
    """
    Train the diffusion model at a fixed gradient checkpointing resolution.
    Args:
        resolution_scale: Patch size scale for model/training (default: GRADIENT_RES_SCALE)
        continue_training: Whether to continue from checkpoint
    """
    # Always use full image size for sampling/generation
    full_img_size = int(IMG_SIZE * RESOLUTION_SCALE)
    patch_size = int(IMG_SIZE * resolution_scale)
    print(f"Training with patch size: {patch_size}x{patch_size} (scale={resolution_scale}), full image size: {full_img_size}x{full_img_size} (scale={RESOLUTION_SCALE})")
    model = SimpleUNet(resolution_scale=RESOLUTION_SCALE).to(DEVICE)
    # Enable gradient checkpointing for memory savings
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    timesteps = 500
    betas = linear_beta_schedule(timesteps).to(DEVICE)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    start_epoch = 0
    best_loss = float('inf')
    patience = 100
    patience_counter = 0
    last_few_losses = []
    checkpoint_path = os.path.join(DATA_DIR, 'diffusion_checkpoint.pth')
    if continue_training and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', best_loss)
            patience_counter = checkpoint.get('patience_counter', 0)
            print(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}, starting fresh")
    def safe_cleanup():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    while True:
        model.train()
        running_loss = 0
        batch_count = 0
        safe_cleanup()
        for batch_idx, imgs in enumerate(train_loader):
            retry_count = 0
            while retry_count < 3:
                try:
                    # Always resize images to full RESOLUTION_SCALE
                    current_full_size = int(IMG_SIZE * RESOLUTION_SCALE)
                    if imgs.shape[2] != current_full_size:
                        resized_imgs = []
                        for img in imgs:
                            resized = torch.nn.functional.interpolate(
                                img.unsqueeze(0),
                                size=(current_full_size, current_full_size),
                                mode='bilinear',
                                align_corners=True
                            )
                            resized_imgs.append(resized.squeeze(0))
                        imgs = torch.stack(resized_imgs)
                    # Random crop patch for each image in the batch
                    patches = []
                    for img in imgs:
                        h, w = img.shape[1:]
                        top = torch.randint(0, h - patch_size + 1, (1,)).item() if h > patch_size else 0
                        left = torch.randint(0, w - patch_size + 1, (1,)).item() if w > patch_size else 0
                        patch = img[:, top:top+patch_size, left:left+patch_size]
                        patches.append(patch)
                    imgs = torch.stack(patches)
                    imgs = imgs.to(DEVICE)
                    t = torch.randint(0, timesteps, (imgs.size(0),), device=DEVICE).long()
                    noisy_imgs, noise = forward_diffusion_sample(imgs, t, betas)
                    optimizer.zero_grad(set_to_none=True)
                    if scaler:
                        with torch.amp.autocast('cuda'):
                            pred = model(noisy_imgs)
                            if pred.shape != noise.shape:
                                target_size = (pred.shape[2], pred.shape[3])
                                noise = torch.nn.functional.interpolate(
                                    noise,
                                    size=target_size,
                                    mode='bilinear',
                                    align_corners=True
                                )
                            assert pred.shape == noise.shape, f"Shape mismatch: pred {pred.shape} vs noise {noise.shape}"
                            loss = criterion(pred, noise)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        pred = model(noisy_imgs)
                        if pred.shape != noise.shape:
                            target_size = (pred.shape[2], pred.shape[3])
                            noise = torch.nn.functional.interpolate(
                                noise,
                                size=target_size,
                                mode='bilinear',
                                align_corners=True
                            )
                        assert pred.shape == noise.shape, f"Shape mismatch: pred {pred.shape} vs noise {noise.shape}"
                        loss = criterion(pred, noise)
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()
                    batch_count += 1
                    if (batch_idx + 1) % 10 == 0:
                        print(f"Epoch {start_epoch+1} - Batch {batch_idx+1}/{len(train_loader)}, "
                              f"Loss: {running_loss/batch_count:.4f}, Patch: {patch_size}, Full: {current_full_size}")
                        if DEVICE.type == 'cuda':
                            torch.cuda.empty_cache()
                    break
                except RuntimeError as e:
                    if ("CUDNN_STATUS_EXECUTION_FAILED" in str(e)) or ("CUDA out of memory" in str(e)):
                        print(f"cuDNN/CUDA error detected on batch {batch_idx}, retry {retry_count+1}/3. Attempting recovery...")
                        clean_memory()
                        torch.cuda.empty_cache()
                        retry_count += 1
                        time.sleep(2)
                        continue
                    else:
                        raise
                except Exception as e:
                    print(f"Non-CUDA error on batch {batch_idx}: {e}")
                    raise
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {start_epoch+1} completed, Avg Loss: {avg_loss:.4f}, Patch: {patch_size}, Full: {current_full_size}")
        last_few_losses.append(avg_loss)
        if len(last_few_losses) > 10:
            last_few_losses.pop(0)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            try:
                torch.save({
                    'state_dict': model.state_dict(),
                    'resolution_scale': RESOLUTION_SCALE,
                    'config': model.config
                }, os.path.join(DATA_DIR, 'best_diffusion_model.pth'))
                print(f"New best model saved with loss: {best_loss:.4f}")
            except Exception as e:
                print(f"Failed to save best model: {e}")
        else:
            patience_counter += 1
            print(f"No improvement: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"No improvement for {patience} epochs! Stopping training.")
                break
        if (start_epoch+1) % 5 == 0:
            try:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': start_epoch,
                    'best_loss': best_loss,
                    'patience_counter': patience_counter,
                    'resolution_scale': RESOLUTION_SCALE
                }, checkpoint_path)
                ckpt_file = os.path.join(DATA_DIR, f'diffusion_model_epoch{start_epoch+1}_res{RESOLUTION_SCALE}.pth')
                torch.save({
                    'state_dict': model.state_dict(),
                    'resolution_scale': RESOLUTION_SCALE,
                    'config': model.config
                }, ckpt_file)
                print(f"Saved checkpoint at epoch {start_epoch+1}")
                ckpts = sorted(glob.glob(os.path.join(DATA_DIR, 'diffusion_model_epoch*.pth')), 
                               key=os.path.getmtime, reverse=True)
                for old_ckpt in ckpts[3:]:
                    try:
                        os.remove(old_ckpt)
                    except Exception as e:
                        print(f"Failed to remove old checkpoint {old_ckpt}: {e}")
            except Exception as e:
                print(f"Error during checkpoint saving: {e}")
        safe_cleanup()
        start_epoch += 1
    print("Training complete.")

# --- IMAGE SAMPLING FUNCTION WITH RESOLUTION SUPPORT ---
def sample_image(model_path=None, out_path=None, steps=500, use_cpu_fallback=False, 
                resolution_scale=None, upscale_factor=1.0):
    """
    Sample an image from the diffusion model with resolution control.
    
    Args:
        model_path: Path to the model weights
        out_path: Path to save the generated image
        steps: Number of diffusion steps (lower = faster but lower quality)
        use_cpu_fallback: If True, force CPU usage for sampling (slower but more reliable)
        resolution_scale: Override model's resolution scale (None = use model's native resolution)
        upscale_factor: Further upscaling factor to apply to output (1.0 = no upscaling)
    """
    # Clear GPU memory before sampling
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Determine device for sampling
    sampling_device = torch.device('cpu') if use_cpu_fallback else DEVICE
    print(f"Sampling using device: {sampling_device}")
    
    try:
        # Load model with proper error handling
        if model_path is None:
            model_path = os.path.join(DATA_DIR, 'best_diffusion_model.pth')
            
        # Check if the file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
        
        # Load model with resolution information
        model_data = torch.load(model_path, map_location='cpu')
        
        # Handle different saved formats (backward compatibility)
        if isinstance(model_data, dict) and 'state_dict' in model_data:
            # New format with metadata
            model_state = model_data['state_dict']
            native_resolution = model_data.get('resolution_scale', 1.0)
            config = model_data.get('config', {'resolution_scale': native_resolution})
        else:
            # Old format (just state dict)
            model_state = model_data
            native_resolution = 1.0
            config = {'resolution_scale': native_resolution}
        
        # Determine final resolution scale
        final_resolution = resolution_scale if resolution_scale is not None else native_resolution
        print(f"Model's native resolution scale: {native_resolution}")
        print(f"Using resolution scale: {final_resolution}")
        
        # Create model with appropriate resolution
        model = SimpleUNet(resolution_scale=final_resolution)
        
        # If we need to change resolution from the native model resolution
        if final_resolution != native_resolution and abs(final_resolution - native_resolution) > 0.01:
            print(f"Scaling model from {native_resolution} to {final_resolution}")
            
            # Create temp model with original resolution and load state
            temp_model = SimpleUNet(resolution_scale=native_resolution)
            temp_model.load_state_dict(model_state)
            
            # Scale to target resolution
            model = temp_model.scale_model_resolution(final_resolution)
        else:
            # Load directly
            model.load_state_dict(model_state)
        
        # Move to appropriate device
        model = model.to(sampling_device)
        model.eval()
        
        # Calculate image size based on model resolution
        base_img_size = IMG_SIZE
        actual_img_size = int(base_img_size * final_resolution)
        
        # Adjust if GPU memory is limited
        if sampling_device.type == 'cuda':
            # Check available memory
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_mem_gb = free_mem / (1024**3)
            print(f"Available GPU memory: {free_mem_gb:.2f} GB")
            
            # If memory is very tight, reduce image size dynamically
            if free_mem_gb < 2.0 and actual_img_size > 48:
                actual_img_size = 48
                print(f"Limited GPU memory, reducing image size to {actual_img_size}x{actual_img_size}")
            # If memory is somewhat tight, cap at 96
            elif free_mem_gb < 4.0 and actual_img_size > 96:
                actual_img_size = 96
                print(f"Limited GPU memory, reducing image size to {actual_img_size}x{actual_img_size}")
        
        print(f"Generating image at {actual_img_size}x{actual_img_size}")
        
        # Using smaller step count if needed
        actual_steps = steps
        if sampling_device.type == 'cuda':
            if free_mem_gb < 2.0:
                actual_steps = min(steps, 100)
            elif free_mem_gb < 4.0:
                actual_steps = min(steps, 200)
                
        if actual_steps < steps:
            print(f"Reducing diffusion steps to {actual_steps} for better memory efficiency")
            
        # Generate initial random noise on CPU then move to device
        img = torch.randn(1, 3, actual_img_size, actual_img_size, device='cpu').to(sampling_device)
        
        # Pre-compute diffusion parameters on CPU first
        betas = linear_beta_schedule(actual_steps)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / torch.cumprod(1 - betas, dim=0))
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / torch.cumprod(1 - betas, dim=0) - 1)
        
        # Move parameters to device when needed
        betas = betas.to(sampling_device)
        sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod.to(sampling_device)
        sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod.to(sampling_device)
        
        # Sampling loop with better error handling
        print(f"Starting sampling process with {actual_steps} diffusion steps")
        for t in reversed(range(actual_steps)):
            if t % 50 == 0 or t < 10:  # More frequent updates in final steps
                print(f"Sampling step {t}/{actual_steps}")
                
            try:
                t_tensor = torch.full((1,), t, device=sampling_device, dtype=torch.long)
                
                # Free memory before prediction
                if sampling_device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Generate prediction
                with torch.no_grad():
                    pred_noise = model(img)
                    
                # Apply noise based on timestep
                if t > 0:
                    noise = torch.randn_like(img)
                else:
                    noise = torch.zeros_like(img)
                
                # Diffusion step
                img = (img - sqrt_recipm1_alphas_cumprod[t] * pred_noise) / sqrt_recip_alphas_cumprod[t] + betas[t] * noise
                
            except RuntimeError as e:
                print(f"Error at step {t}: {e}")
                print("Attempting to continue with CPU...")
                
                # Try to recover by moving to CPU
                if sampling_device.type == 'cuda':
                    # Move everything to CPU and continue
                    model = model.to('cpu')
                    img = img.to('cpu')
                    betas = betas.to('cpu')
                    sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod.to('cpu')
                    sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod.to('cpu')
                    sampling_device = torch.device('cpu')
                    print("Switched to CPU for sampling")
                else:
                    # If already on CPU and still failing, we have to abort
                    print("Error occurred on CPU, cannot proceed with sampling.")
                    return
        
        # Final processing
        img = (img.clamp(-1, 1) + 1) / 2
        
        # Apply additional upscaling if requested
        if upscale_factor > 1.0:
            # Move to CPU for upscaling to avoid GPU memory issues
            img_cpu = img.cpu()
            
            # Calculate target size
            final_size = int(actual_img_size * upscale_factor)
            print(f"Upscaling result to {final_size}x{final_size}")
            
            # Use interpolate for upscaling
            img_upscaled = torch.nn.functional.interpolate(
                img_cpu, 
                size=(final_size, final_size), 
                mode='bicubic',  # Higher quality interpolation
                align_corners=False
            )
            
            img = img_upscaled
        
        # Set output path
        if out_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(DATA_DIR, f'sampled_axolotl_{timestamp}.png')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            
        # Move to CPU for saving if needed
        if img.device.type != 'cpu':
            img = img.cpu()
            
        # Save the image
        save_image(img, out_path)
        print(f"Sampled image saved to {out_path}")
        
        return out_path
        
    except Exception as e:
        print(f"Unexpected error during sampling: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    import sys
    import argparse
    import traceback  # For detailed error reporting
    import os
    
    # Enable CUDA error debugging (for misaligned address and memory issues)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
      # Setup command line arguments
    parser = argparse.ArgumentParser(description='Axolotl Diffusion Model with Progressive Scaling')
    parser.add_argument('mode', type=str, nargs='?', default='train', choices=['train', 'sample', 'scale'], 
                        help='Mode to run in: train, sample, or scale (converts model to new resolution)')
    
    # Training arguments
    parser.add_argument('--resolution', type=float, default=GRADIENT_RES_SCALE,
                        help='Resolution scale factor for training (default: GRADIENT_RES_SCALE, e.g. 0.5). Use a higher value only if you have enough VRAM. For sampling/generation, use RESOLUTION_SCALE.')
    parser.add_argument('--fresh', action='store_true',
                        help='Start training from scratch (ignore checkpoints)')
                        
    # Sampling arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save generated sample image')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model weights for sampling')
    parser.add_argument('--steps', type=int, default=100, 
                        help='Number of diffusion steps for sampling')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU mode for sampling (more stable but slower)')
    parser.add_argument('--upscale', type=float, default=1.0,
                        help='Additional upscaling factor for final image')
                        
    # Model scaling arguments
    parser.add_argument('--target-scale', type=float, 
                        help='Target resolution scale when using --mode=scale')
    parser.add_argument('--output-model', type=str,
                        help='Path to save scaled model when using --mode=scale')
                        
    # General arguments
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with extra logging')
    
    args = parser.parse_args()
    
    # Print system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print_gpu_memory_status()
    else:
        print("CUDA is not available. Using CPU.")
        
    print(f"Using device: {DEVICE}")
    
    # Ensure data directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True) 
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Run the appropriate mode
    try:
        if args.mode == 'sample':
            print(f"Sampling axolotl image with {args.steps} steps at resolution {args.resolution}")
            # Clean memory before sampling
            clean_memory()
            sample_image(
                model_path=args.model, 
                out_path=args.output, 
                steps=args.steps, 
                use_cpu_fallback=args.cpu,
                resolution_scale=args.resolution,
                upscale_factor=args.upscale
            )
        
        elif args.mode == 'scale':
            if not args.target_scale or not args.model or not args.output_model:
                print("Error: --target-scale, --model and --output-model are required for scale mode")
                sys.exit(1)
                
            print(f"Scaling model from {args.model} to resolution {args.target_scale}")
            
            # Load model
            try:
                model_data = torch.load(args.model, map_location='cpu')
                
                # Handle different saved formats
                if isinstance(model_data, dict) and 'state_dict' in model_data:
                    # New format with metadata
                    model_state = model_data['state_dict']
                    native_resolution = model_data.get('resolution_scale', 1.0)
                else:
                    # Old format (just state dict)
                    model_state = model_data
                    native_resolution = 1.0
                
                # Create model with original resolution
                original_model = SimpleUNet(resolution_scale=native_resolution)
                original_model.load_state_dict(model_state)
                
                # Scale to new resolution
                print(f"Scaling from {native_resolution} to {args.target_scale}")
                new_model = original_model.scale_model_resolution(args.target_scale)
                
                # Save with metadata
                torch.save({
                    'state_dict': new_model.state_dict(),
                    'resolution_scale': args.target_scale,
                    'config': new_model.config
                }, args.output_model)
                
                print(f"Successfully saved scaled model to {args.output_model}")
                
            except Exception as e:
                print(f"Error scaling model: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
        
        else:  # Train mode
            print(f"Starting diffusion model training at resolution {args.resolution}")
                
            # Clean memory before training
            clean_memory()
            train(
                resolution_scale=args.resolution,
                continue_training=not args.fresh
            )
            
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print("=" * 80)
            print("CUDA OUT OF MEMORY ERROR")
            print("Try these solutions:")
            print("1. Use --cpu flag to force CPU execution")
            print("2. Use --steps 50 to reduce diffusion steps")
            print("3. Reduce --resolution to a lower value (e.g. 0.75)")
            print("4. Free up GPU memory by closing other applications")
            print("=" * 80)
        elif 'illegal memory access' in str(e):
            print("=" * 80)
            print("CUDA ILLEGAL MEMORY ACCESS ERROR")
            print("This often happens due to bugs in CUDA kernels or memory corruption.")
            print("Try these solutions:")
            print("1. Use --cpu flag to force CPU execution")
            print("2. Update your GPU drivers")
            print("3. Try a smaller model with --resolution 0.5")
            print("=" * 80)
        else:
            print(f"Runtime error: {e}")
        
        # Print detailed error information in debug mode
        if args.debug:
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
