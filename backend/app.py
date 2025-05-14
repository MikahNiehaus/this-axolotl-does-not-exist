from flask import Flask, jsonify
from models.generator import generate_axolotl_image

app = Flask(__name__)

@app.route('/generate', methods=['GET'])
def generate_image():
    image = generate_axolotl_image()
    return jsonify({'image': image})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)