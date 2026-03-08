from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, 'models')

model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
model_results = joblib.load(os.path.join(MODEL_DIR, 'model_results.pkl'))
best_model_name = joblib.load(os.path.join(MODEL_DIR, 'best_model_name.pkl'))

CHOICES = {
    'Month': ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
    'DayOfWeek': ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
    'Make': ['Accura','BMW','Chevrolet','Dodge','Ferrari','Ford','Honda','Jaguar','Lexus',
             'Mazda','Mecedes','Mercury','Nisson','Pontiac','Porche','Saab','Saturn','Toyota','VW'],
    'AccidentArea': ['Urban','Rural'],
    'DayOfWeekClaimed': ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','0'],
    'MonthClaimed': ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','0'],
    'Sex': ['Male','Female'],
    'MaritalStatus': ['Single','Married','Divorced','Widow'],
    'Fault': ['Policy Holder','Third Party'],
    'PolicyType': ['Sedan - All Perils','Sedan - Collision','Sedan - Liability',
                   'Sport - All Perils','Sport - Collision','Sport - Liability',
                   'Utility - All Perils','Utility - Collision','Utility - Liability'],
    'VehicleCategory': ['Sedan','Sport','Utility'],
    'VehiclePrice': ['less than 20000','20000 to 29000','30000 to 39000',
                     '40000 to 59000','60000 to 69000','more than 69000'],
    'Days_Policy_Accident': ['none','1 to 7','8 to 15','15 to 30','more than 30'],
    'Days_Policy_Claim': ['none','8 to 15','15 to 30','more than 30'],
    'PastNumberOfClaims': ['none','1','2 to 4','more than 4'],
    'AgeOfVehicle': ['new','2 years','3 years','4 years','5 years','6 years','7 years','more than 7'],
    'AgeOfPolicyHolder': ['16 to 17','18 to 20','21 to 25','26 to 30','31 to 35',
                          '36 to 40','41 to 50','51 to 65','over 65'],
    'PoliceReportFiled': ['Yes','No'],
    'WitnessPresent': ['Yes','No'],
    'AgentType': ['External','Internal'],
    'NumberOfSuppliments': ['none','1 to 2','3 to 5','more than 5'],
    'AddressChange_Claim': ['no change','under 6 months','1 year','2 to 3 years','4 to 8 years'],
    'NumberOfCars': ['1 vehicle','2 vehicles','3 to 4','5 to 8','more than 8'],
    'BasePolicy': ['Liability','Collision','All Perils'],
}

@app.route('/')
def index():
    return render_template('index.html',
                           model_results=model_results,
                           best_model_name=best_model_name,
                           choices=CHOICES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        for col, le in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except:
                    input_df[col] = 0

        for col in feature_names:
            try:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
            except:
                input_df[col] = 0

        X = input_df[feature_names].values
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

        return jsonify({
            'prediction': int(prediction),
            'label': 'FRAUD' if prediction == 1 else 'NOT FRAUD',
            'confidence': round(float(max(proba)) * 100, 2),
            'fraud_prob': round(float(proba[1]) * 100, 2),
            'not_fraud_prob': round(float(proba[0]) * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-stats')
def model_stats():
    return jsonify({
        'results': {k: round(v*100, 2) for k, v in model_results.items()},
        'best': best_model_name,
        'best_accuracy': round(model_results[best_model_name]*100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
