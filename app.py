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
        # Get all form data
        form_data = dict(request.form)

        # Extract patient name (not used in prediction)
        patient_name = form_data.pop("patient_name", "User")

        # Convert remaining values to float
        values = [float(val) for val in form_data.values()]
        features = np.array([values])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        # Make result human-readable
        result = "✅ No Heart Disease" if prediction == 0 else "⚠️ Heart Disease Detected"

        return render_template('index.html', prediction_text=f"{patient_name}, your result: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# Optional JSON API version
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
