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
            FULL_MODEL_PATH = os.path.join(DATA_DIR, 'gan_full_model.pth')
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
                def load_model(self):
                    if os.path.exists(FULL_MODEL_PATH):
                        log(f"Loading full model from: {FULL_MODEL_PATH}")
                        checkpoint = torch.load(FULL_MODEL_PATH, map_location=self.device)
                        log(f"Checkpoint keys: {list(checkpoint.keys())}")
                        self.G.load_state_dict(checkpoint['G'])
                        log("Loaded model for generating sample")
                    else:
                        log("ERROR: Full model file (gan_full_model.pth) not found. Aborting.")
                        raise FileNotFoundError("Full model file (gan_full_model.pth) not found.")
                def generate_sample(self):
                    log("Calling G.eval() and generating image...")
                    self.G.eval()
                    with torch.no_grad():
                        fake = self.G(self.fixed_noise[:1]).detach().cpu()
                        log(f"Fake image tensor shape: {fake.shape}")
                        buffer = io.BytesIO()
                        save_image(fake, buffer, format="PNG", normalize=True)
                        buffer.seek(0)
                        log("Image saved to buffer and ready to encode.")
                        return buffer.getvalue()
            trainer = GANTrainer(img_size=IMG_SIZE, z_dim=Z_DIM, device=DEVICE)
            trainer.load_model()
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