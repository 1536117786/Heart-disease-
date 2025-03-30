from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os
import pandas as pd
from datetime import datetime

# Load model and scaler
model = pickle.load(open('heart_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for form prediction and result page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = dict(request.form)

        # Extract patient name and language
        patient_name = form_data.pop("patient_name", "User")
        language = form_data.pop("language", "en")

        # Convert remaining form values to float
        input_values = [float(val) for val in form_data.values()]
        features = np.array([input_values])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        result_text = "✅ No Heart Disease" if prediction == 0 else "⚠️ Heart Disease Detected"

        # Log predictions to CSV
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient_name": patient_name,
            **form_data,
            "prediction": int(prediction),
            "result": result_text
        }

        df_log = pd.DataFrame([log_entry])
        if os.path.exists('predictions_log.csv'):
            df_log.to_csv('predictions_log.csv', mode='a', header=False, index=False)
        else:
            df_log.to_csv('predictions_log.csv', mode='w', header=True, index=False)

        # Render result page
        return render_template(
            'result.html',
            patient_name=patient_name,
            prediction_text=result_text,
            prediction=int(prediction),
            language=language
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# Optional: API Endpoint for JSON requests
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
