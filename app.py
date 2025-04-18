from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from recommendation_system import categorize_risk, get_treatment_recommendation

app = Flask(__name__)

# Check if models exist, otherwise show appropriate message
model_path = 'models/best_model.pkl'
scaler_path = 'models/scaler.pkl'
imputer_path = 'models/imputer.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(imputer_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    model_loaded = True
else:
    model_loaded = False


@app.route('/')
def home():
    if model_loaded:
        return render_template('index.html')
    else:
        return render_template('model_not_found.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})

    try:
        # Get form data
        features = {
            'Pregnancies': float(request.form['pregnancies']),
            'Glucose': float(request.form['glucose']),
            'BloodPressure': float(request.form['bloodpressure']),
            'SkinThickness': float(request.form['skinthickness']),
            'Insulin': float(request.form['insulin']),
            'BMI': float(request.form['bmi']),
            'DiabetesPedigreeFunction': float(request.form['diabetespedigreefunction']),
            'Age': float(request.form['age'])
        }

        # Create a dataframe
        patient_data = pd.DataFrame([features])

        # Handle missing values (replace 0 with NaN for certain columns)
        zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for column in zero_columns:
            patient_data[column] = patient_data[column].replace(0, np.nan)

        # Impute missing values
        patient_data_imputed = pd.DataFrame(
            imputer.transform(patient_data),
            columns=patient_data.columns
        )

        # Scale the data
        patient_data_scaled = scaler.transform(patient_data_imputed)

        # Make prediction
        risk_probability = model.predict_proba(patient_data_scaled)[0, 1]
        risk_category = categorize_risk(risk_probability)

        # Get recommendations
        recommendations = get_treatment_recommendation(risk_category, features)

        # Return the results
        return jsonify({
            'probability': float(risk_probability),
            'risk_category': risk_category,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, port=8080)  # Use port 8080 instead of default 5000