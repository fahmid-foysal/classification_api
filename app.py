from flask import Flask, request, jsonify
import base64
from single_predictor import predict_single
from bulk_predictor import predict_bulk

app = Flask(__name__)


@app.route('/predict/single', methods=['POST'])
def predict_single_route():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file found in request"}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        result = predict_single(image_base64)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict/bulk', methods=['POST'])
def predict_bulk_route():
    try:
        if 'images' not in request.files:
            return jsonify({"error": "No image files found in request"}), 400

        files = request.files.getlist('images')
        image_base64_list = []

        for file in files:
            image_bytes = file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image_base64_list.append(image_base64)

        result = predict_bulk(image_base64_list)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
