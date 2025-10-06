from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

def load_model():
    
    model = joblib.load('ckd.pkl')
    return model


model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        specific_gravity = float(request.form['specific_gravity'])
        hypertension = int(request.form['hypertension'])
        haemoglobin = float(request.form['haemoglobin'])
        diabetes_mellitus = int(request.form['diabetes_mellitus'])
        albumin = float(request.form['albumin'])
        serum_creatinine = float(request.form['serum_creatinine'])
        aanemia = int(request.form['aanemia'])
        pus_cell = int(request.form['pus_cell'])
        
        input_data = np.array([[specific_gravity, hypertension, haemoglobin, diabetes_mellitus,
                                albumin, serum_creatinine, aanemia, pus_cell]])
        
        prediction_result = model.predict(input_data)
        result_text = "CKD" if prediction_result == 1 else "Not CKD"
        
        return render_template('result.html', prediction_result=result_text)
    
    except Exception as e:
        error_message = "An error occurred: " + str(e)
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
