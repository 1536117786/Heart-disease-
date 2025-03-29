# 💓 Heart Disease Prediction API

An end-to-end Machine Learning project that predicts the likelihood of heart disease using clinical parameters. The project includes model training, API development with Flask, and deployment using Render.

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early detection can save lives. This project provides a simple, lightweight API to predict heart disease based on patient health metrics.

---

## 🚀 Tech Stack

- **Language**: Python
- **Machine Learning**: scikit-learn
- **Web Framework**: Flask
- **Deployment**: Render
- **Data Handling**: NumPy, Pandas
- **Model Serialization**: Pickle

---

## 🧠 Features Used for Prediction

| Feature       | Description                          |
|---------------|--------------------------------------|
| `age`         | Age in years                         |
| `sex`         | Sex (1 = male; 0 = female)           |
| `cp`          | Chest pain type (0–3)                |
| `trestbps`    | Resting blood pressure (mm Hg)       |
| `chol`        | Serum cholesterol (mg/dl)            |
| `fbs`         | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| `restecg`     | Resting electrocardiographic results |
| `thalach`     | Maximum heart rate achieved          |
| `exang`       | Exercise-induced angina (1 = yes; 0 = no) |
| `oldpeak`     | ST depression induced by exercise    |
| `slope`       | Slope of the peak exercise ST segment |
| `ca`          | Number of major vessels (0–3)        |
| `thal`        | Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect) |

---

## 📦 Project Structure

heart-disease-predictor/
│
├── model/
│   ├── heart_model.pkl          # Trained ML model
│   └── scaler.pkl               # Feature scaler
│
├── static/                      # (Optional) For CSS/images if you build a frontend
├── templates/                   # (Optional) HTML templates for Flask frontend
│
├── app.py                       # Main Flask API app
├── test_api.py                  # Test script for API prediction (requests code)
├── requirements.txt             # All dependencies
├── .gitignore                   # Ignore unnecessary files/folders
├── Procfile                     # (Optional for Heroku-style deployment, not Render)
├── README.md                    # Full project documentation
└── utils/
    └── preprocess.py            # (Optional) Helper functions (e.g., feature processing)



---

## ⚙️ How It Works

### 1. Model Training
- Used a classification model (e.g., Logistic Regression / Random Forest).
- Dataset: [Kaggle Dataset] https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci
- Preprocessed and trained the model with `scikit-learn`.
- Saved model using `pickle`.

### 2. API Development
- Built a Flask API to accept POST requests.
- Input: JSON object with 13 clinical features.
- Output: Prediction (0 = No heart disease, 1 = Heart disease).

### 3. Deployment
- Deployed the API to **Render**.
- Public URL: [https://heart-disease-api-tfyk.onrender.com]

---

## 🔎 Example Prediction (Python Client)

```python
import requests

url = 'https://heart-disease-api-tfyk.onrender.com/predict'

input_data = {
    "age": 45,
    "sex": 1,
    "cp": 0,
    "trestbps": 130,
    "chol": 230,
    "fbs": 0,
    "restecg": 1,
    "thalach": 180,
    "exang": 0,
    "oldpeak": 0.5,
    "slope": 2,
    "ca": 0,
    "thal": 2
}

response = requests.post(url, json=input_data)

if response.status_code == 200:
    print("✅ Prediction:", response.json()['prediction'])
else:
    print("❌ Error:", response.status_code)
    print(response.text)

---

## Future Improvements

Add frontend for user input.

Model explainability using SHAP.

Logging and monitoring for API.

Containerize with Docker.

CI/CD pipeline for automatic deployment.

---

👨‍💻 Author
Pavan Kumar Dandi
🎓 Master’s in Applied Statistics & Data Analytics – Central Michigan University
🎯 Aspiring Machine Learning Engineer
📫 www.linkedin.com/in/pavankumardandi
