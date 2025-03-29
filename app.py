from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

model = pickle.load(open('heart_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Heart Disease Prediction API is Live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([list(data.values())])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
