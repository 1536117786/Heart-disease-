from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

# Load model and scaler
model = pickle.load(open('heart_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

# Route for homepage with form
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction via web form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values and convert to float
        values = [float(request.form.get(key)) for key in request.form]
        features = np.array([values])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        result = "✅ No Heart Disease" if prediction == 0 else "⚠️ Heart Disease Detected"
        return render_template('index.html', prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# Optional JSON API version (same as before)
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    features = np.array([list(data.values())])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
