from flask import Flask, jsonify, send_file
import io
import base64
from models.generator import AxolotlGenerator
import numpy as np
from PIL import Image
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

gan = AxolotlGenerator()

class AxolotlImageAPI:
    @staticmethod
    def generate_single_image():
        # Use the full GAN sample process: generate, post-process, and send the image as base64 (do not save to disk)
        try:
            import torch
            from torchvision.utils import make_grid
            from models.gan_modules import Generator
            import numpy as np
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint_path = os.path.join(os.path.dirname(__file__), 'data', 'gan_checkpoint.pth')
            img_size = 128  # Use the highest size your GAN supports for best quality
            z_dim = 100
            G = Generator(z_dim=z_dim, img_channels=3, img_size=img_size).to(device)
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                G.load_state_dict(checkpoint['G'])
            G.eval()
            with torch.no_grad():
                noise = torch.randn(1, z_dim, 1, 1, device=device)
                fake = G(noise).detach().cpu()
                # Post-process: denormalize and convert to numpy
                img_tensor = fake[0]
                img_tensor = (img_tensor + 1) / 2  # [-1,1] -> [0,1]
                img_np = img_tensor.mul(255).clamp(0,255).byte().numpy()
                img_np = np.transpose(img_np, (1,2,0))  # CHW to HWC
                img = Image.fromarray(img_np)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
            img_str = base64.b64encode(img_bytes).decode("utf-8")
            return img_str
        except Exception as e:
            # fallback to Keras generator if torch fails
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

@app.route('/generate', methods=['GET'])
def generate_image():
    img_b64 = AxolotlImageAPI.generate_single_image()
    return jsonify({'image': img_b64})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)