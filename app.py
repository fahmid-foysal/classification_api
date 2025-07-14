from flask import Flask, request, jsonify
from single_predictor import predict_single
from bulk_predictor import predict_bulk

app = Flask(__name__)

@app.route('/predict/single', methods=['POST'])
def predict_single_route():
    try:
        data = request.get_json(force=True)
        result = predict_single(data['image_base64'])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict/bulk', methods=['POST'])
def predict_bulk_route():
    try:
        data = request.get_json(force=True)
        result = predict_bulk(data['image_base64'])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ✅ Required for cPanel WSGI
app = app
