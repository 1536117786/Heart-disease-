
# 🫀 Heart Disease Prediction Project — ML + Flask API + Deployment

This project focuses on predicting the presence of heart disease using machine learning models. It also demonstrates how to build, evaluate, and deploy a model as a live REST API using Flask and Render.

---

## 🔍 Project Overview

Heart disease is one of the leading causes of death globally. Early detection can save lives. This project uses a dataset of clinical features to build a classification model that predicts whether a patient has heart disease.

---

## 📊 Dataset & EDA

The dataset used contains the following features:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting electrocardiographic results
- Maximum heart rate
- Exercise-induced angina
- ST depression
- Slope of ST
- Number of vessels observed
- Thallium stress test result

EDA steps included:
- Distribution plots for each feature
- KDE plots comparing features with target
- Crosstabs to explore categorical variable impact
- Min, Max, and Average age of patients with heart disease

---

## 🤖 Model Building and Evaluation

We trained 5 different ML models and compared their performance:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVC)
- Random Forest Classifier
- Decision Tree Classifier

Each model was:
- Trained on a scaled dataset
- Evaluated using accuracy score
- Compared using a bar chart

**Final selected model** was the one with highest accuracy.

---

## 🚀 Deployment as Flask API

After selecting the best model, we:

- Saved the model and scaler using `pickle`
- Built a Flask REST API (`app.py`)
- Created a POST endpoint `/predict` to accept input and return predictions
- Deployed the app live using **Render.com**

---

## 🌐 Live Demo

**API Base URL**: [https://your-app-name.onrender.com](https://your-app-name.onrender.com)  
✅ Replace with your actual Render link.

---

## 📥 Sample JSON Input

```json
{
  "age": 58,
  "sex": 1,
  "cp": 2,
  "trestbps": 130,
  "chol": 230,
  "fbs": 0,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
```

### 📤 JSON Output

```json
{
  "prediction": 1
}
```

---

## 📁 Folder Structure

```
heart-disease-api/
│
├── app.py                 # Flask API code
├── heart_model.pkl        # Trained ML model
├── scaler.pkl             # StandardScaler
├── requirements.txt       # Dependency file
├── README.md              # Project documentation
```

---

## 🧰 Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn (EDA)
- Flask (API)
- Render (Deployment)
- Git & GitHub (Version Control)

---

## 👨‍💻 Author

**Pavan Kumar**  
📧 [your-email@example.com]  
🔗 [LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

### 💬 Feel free to connect or reach out for collaborations!
