from flask import Flask, jsonify, send_file
import io
import base64
from models.generator import AxolotlGenerator
import numpy as np
from PIL import Image
from flask_cors import CORS
import os
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Ensure directories exist
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SAMPLE_DIR = os.path.join(DATA_DIR, 'gan_samples')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

gan = AxolotlGenerator()

class AxolotlImageAPI:
    @staticmethod
    def generate_single_image():
        logs = []
        def log(msg):
            print(msg)
            logs.append(str(msg))
        try:
            log('--- [BACKEND] Starting GAN image generation ---')
            import torch
            import torch.nn as nn
            from torchvision.utils import save_image
            from models.gan_modules import Generator, Discriminator
            import numpy as np
            log(f"torch version: {torch.__version__}")
            log(f"torchvision version: {__import__('torchvision').__version__}")
            log(f"numpy version: {np.__version__}")
            DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
            SAMPLE_DIR = os.path.join(DATA_DIR, 'gan_samples')
            CHECKPOINT_PATH = os.path.join(DATA_DIR, 'gan_checkpoint.pth')
            IMG_SIZE = 32
            Z_DIM = 100
            DEVICE = torch.device('cpu')
            log(f"DEVICE: {DEVICE}")
            class GANTrainer:
                def __init__(self, img_size, z_dim, device):
                    self.img_size = img_size
                    self.z_dim = z_dim
                    self.device = device
                    self.G = Generator(z_dim=z_dim, img_channels=3, img_size=img_size).to(device)
                    self.fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)
                def load_checkpoint(self):
                    log(f"Looking for checkpoint at: {CHECKPOINT_PATH}")
                    if os.path.exists(CHECKPOINT_PATH):
                        try:
                            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
                            log(f"Successfully loaded checkpoint file")
                        except Exception as e:
                            log(f"Error loading checkpoint: {str(e)}")
                            return False
                        
                        # Check if checkpoint is a valid dictionary
                        if not isinstance(checkpoint, dict):
                            log(f"Error: Checkpoint is not a dictionary, got {type(checkpoint)}")
                            return False
                        
                        log(f"Checkpoint keys: {list(checkpoint.keys())}")
                        
                        # Check for different possible key formats
                        # Handle different model checkpoint key formats
                        if 'G' in checkpoint:
                            # This is the expected format from train_gan.py
                            try:
                                self.G.load_state_dict(checkpoint['G'])
                                log("Loaded 'G' key from checkpoint")
                                return True
                            except Exception as e:
                                log(f"Error loading 'G' state dict: {str(e)}. Trying to adapt keys...")
                                try:
                                    # Try to adapt the state dict to match the model
                                    state_dict = checkpoint['G']
                                    # Clean keys by removing module. prefix if it exists
                                    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                                    # Try a flexible loading approach
                                    model_dict = self.G.state_dict()
                                    # Filter out keys that don't match the model
                                    filtered_dict = {k: v for k, v in clean_state_dict.items() if k in model_dict}
                                    missing = [k for k in model_dict.keys() if k not in filtered_dict]
                                    if missing:
                                        log(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
                                    self.G.load_state_dict(filtered_dict, strict=False)
                                    log("Loaded partial model from 'G' key with flexible loading")
                                    return True
                                except Exception as e2:
                                    log(f"Error with flexible loading: {str(e2)}")
                        elif 'model_G' in checkpoint:
                            try:
                                self.G.load_state_dict(checkpoint['model_G'])
                                log("Loaded 'model_G' key from checkpoint")
                                return True
                            except Exception as e:
                                log(f"Error loading 'model_G' state dict: {str(e)}")
                        elif 'generator' in checkpoint:
                            try:
                                self.G.load_state_dict(checkpoint['generator'])
                                log("Loaded 'generator' key from checkpoint")
                                return True
                            except Exception as e:
                                log(f"Error loading 'generator' state dict: {str(e)}")
                        elif 'state_dict' in checkpoint:
                            try:
                                self.G.load_state_dict(checkpoint['state_dict'])
                                log("Loaded 'state_dict' key from checkpoint")
                                return True
                            except Exception as e:
                                log(f"Error loading 'state_dict': {str(e)}")
                        
                        # Last resort: try all keys that might contain state dictionaries
                        for key in checkpoint.keys():
                            if isinstance(checkpoint[key], dict) and 'weight' in str(checkpoint[key].keys()):
                                try:
                                    self.G.load_state_dict(checkpoint[key], strict=False)
                                    log(f"Loaded weights from key '{key}' as best effort")
                                    return True
                                except Exception as e:
                                    log(f"Failed to load weights from key '{key}': {str(e)}")
                        
                        # If we reach here, no suitable keys were found
                        log(f"Error: Could not find compatible model weights in checkpoint")
                        return False
                    else:
                        log("No checkpoint found, using untrained generator")
                        return False
                def generate_sample(self):
                    log("Calling G.eval() and generating image...")
                    self.G.eval()
                    try:
                        with torch.no_grad():
                            try:
                                fake = self.G(self.fixed_noise[:1]).detach().cpu()
                                log(f"Fake image tensor shape: {fake.shape}")
                                
                                # Handle potential issues with the generated tensor
                                if torch.isnan(fake).any() or torch.isinf(fake).any():
                                    log("Warning: Generated image contains NaN or Inf values. Clamping...")
                                    fake = torch.nan_to_num(fake, nan=0.0, posinf=1.0, neginf=-1.0)
                                    fake = torch.clamp(fake, -1.0, 1.0)
                                    
                                buffer = io.BytesIO()
                                save_image(fake, buffer, format="PNG", normalize=True)
                                buffer.seek(0)
                                log("Image saved to buffer and ready to encode.")
                                return buffer.getvalue()
                            except RuntimeError as e:
                                if 'out of memory' in str(e).lower():
                                    log("CUDA out of memory error. Trying with smaller batch...")
                                    # Clear cache and try with single sample
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    # Recreate noise with smaller size
                                    single_noise = torch.randn(1, self.z_dim, 1, 1, device=self.device)
                                    fake = self.G(single_noise).detach().cpu()
                                    buffer = io.BytesIO()
                                    save_image(fake, buffer, format="PNG", normalize=True)
                                    buffer.seek(0)
                                    log("Image saved to buffer and ready to encode (with memory optimization).")
                                    return buffer.getvalue()
                                else:
                                    raise  # Re-raise if not a memory error
                    except Exception as e:
                        log(f"Error generating image: {str(e)}. Attempting fallback...")
                        # Last resort: generate a very small image
                        try:
                            # Create a tiny random tensor as a placeholder
                            placeholder = torch.rand(1, 3, 32, 32) * 2 - 1  # Range [-1, 1]
                            buffer = io.BytesIO()
                            save_image(placeholder, buffer, format="PNG", normalize=True)
                            buffer.seek(0)
                            log("Used fallback random image instead of generator output.")
                            return buffer.getvalue()
                        except Exception as e2:
                            log(f"Fallback also failed: {str(e2)}")
                            raise
            trainer = GANTrainer(img_size=IMG_SIZE, z_dim=Z_DIM, device=DEVICE)
            trainer.load_checkpoint()
            img_bytes = trainer.generate_sample()
            log("Image generation complete, encoding to base64...")
            img_str = base64.b64encode(img_bytes).decode("utf-8")
            log("Image base64 encoding complete.")
            return img_str, logs
        except Exception as e:
            log(f"PyTorch GAN failed: {str(e)}")
            return None, logs

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': str(time.time())
    })

@app.route('/generate', methods=['GET'])
def generate_image():
    img_b64, logs = AxolotlImageAPI.generate_single_image()
    return jsonify({'image': img_b64, 'logs': logs})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)