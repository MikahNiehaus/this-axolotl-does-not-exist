from flask import Flask, jsonify, send_file
import io
import base64
from models.generator import AxolotlGenerator
import numpy as np
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

gan = AxolotlGenerator()

class AxolotlImageAPI:
    @staticmethod
    def generate_single_image():
        # Generate a single image from GAN
        noise = np.random.normal(0, 1, (1, gan.input_shape[0]))
        img_arr = gan.generate_image(noise)
        # Flatten batch if needed
        if isinstance(img_arr, list) or len(img_arr.shape) == 4:
            img_arr = img_arr[0]
        # Rescale from [-1,1] to [0,255] and convert to uint8
        img_arr = ((img_arr + 1) * 127.5).clip(0, 255).astype(np.uint8)
        # Try to reshape to (H, W, 3) if possible, else fallback to 1D square RGB
        try:
            img_arr = img_arr.reshape(gan.img_shape)
        except Exception:
            # Fallback: flatten and create a square RGB image from the data
            flat = img_arr.flatten()
            # Calculate side length for square
            side = int(np.ceil(np.sqrt(flat.size / 3)))
            # Pad if needed
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