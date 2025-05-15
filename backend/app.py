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
        # Use the exact same GAN sample command from train_gan.py for best quality
        try:
            import torch
            import torch.nn as nn
            from torchvision.utils import save_image
            from models.gan_modules import Generator, Discriminator
            import numpy as np
            
            # Match the exact constants from train_gan.py
            DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
            SAMPLE_DIR = os.path.join(DATA_DIR, 'gan_samples')
            CHECKPOINT_PATH = os.path.join(DATA_DIR, 'gan_checkpoint.pth')
            IMG_SIZE = 32  # Match the IMG_SIZE in train_gan.py
            Z_DIM = 100
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create the GANTrainer instance exactly like train_gan.py
            class GANTrainer:
                def __init__(self, img_size, z_dim, device):
                    self.img_size = img_size
                    self.z_dim = z_dim
                    self.device = device
                    self.G = Generator(z_dim=z_dim, img_channels=3, img_size=img_size).to(device)
                    self.fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)
                    
                def load_checkpoint(self):
                    if os.path.exists(CHECKPOINT_PATH):
                        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
                        self.G.load_state_dict(checkpoint['G'])
                        print("Loaded checkpoint for generating sample")
                    else:
                        print("No checkpoint found, using untrained generator")
                
                def generate_sample(self):
                    self.G.eval()
                    with torch.no_grad():
                        # Use the same fixed noise as in train_gan.py
                        fake = self.G(self.fixed_noise[:1]).detach().cpu()
                        # Save to memory buffer instead of disk
                        buffer = io.BytesIO()
                        save_image(fake, buffer, format="PNG", normalize=True)
                        buffer.seek(0)
                        return buffer.getvalue()
            
            # Create trainer and generate sample
            trainer = GANTrainer(img_size=IMG_SIZE, z_dim=Z_DIM, device=DEVICE)
            trainer.load_checkpoint()
            img_bytes = trainer.generate_sample()
                
            # Return the image as base64
            img_str = base64.b64encode(img_bytes).decode("utf-8")
            return img_str
        except Exception as e:
            # Fallback to Keras generator if PyTorch fails
            print(f"PyTorch GAN failed, falling back to Keras: {str(e)}")
            noise = np.random.normal(0, 1, (1, gan.input_shape[0]))
            img_arr = gan.generate_image(noise)
            if isinstance(img_arr, list) or len(img_arr.shape) == 4:
                img_arr = img_arr[0]
            img_arr = ((img_arr + 1) * 127.5).clip(0, 255).astype(np.uint8)
            try:
                img_arr = img_arr.reshape(gan.img_shape)
            except Exception:
                flat = img_arr.flatten()
                side = int(np.ceil(np.sqrt(flat.size / 3)))
                pad = side * side * 3 - flat.size
                if pad > 0:
                    flat = np.pad(flat, (0, pad), mode='constant', constant_values=0)
                img_arr = flat.reshape((side, side, 3))
            img = Image.fromarray(img_arr)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': str(time.time())
    })

@app.route('/generate', methods=['GET'])
def generate_image():
    img_b64 = AxolotlImageAPI.generate_single_image()
    return jsonify({'image': img_b64})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)