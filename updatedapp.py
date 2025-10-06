from flask import Flask, render_template, request, Response
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from reportlab.pdfgen import canvas

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
        # Extract input data from the form
        specific_gravity = float(request.form['specific_gravity'])
        hypertension = int(request.form['hypertension'])
        haemoglobin = float(request.form['haemoglobin'])
        diabetes_mellitus = int(request.form['diabetes_mellitus'])
        albumin = float(request.form['albumin'])
        serum_creatinine = float(request.form['serum_creatinine'])
        aanemia = int(request.form['aanemia'])
        pus_cell = int(request.form['pus_cell'])
        
        # Prepare the input data for prediction
        input_data = np.array([[specific_gravity, hypertension, haemoglobin, diabetes_mellitus,
                                albumin, serum_creatinine, aanemia, pus_cell]])
        
        # Perform data preprocessing if needed (e.g., scaling)
        # scaler = StandardScaler()  # Use the same scaler used during training
        # input_data = scaler.transform(input_data)
        
        # Make predictions using the loaded model
        prediction_result = model.predict(input_data)
        
        # Convert the prediction result (1 for CKD, 0 for Not CKD) into a human-readable format
        result_text = "CKD Positive" if prediction_result == 1 else "CKD Negative"
        
        # Render the 'result.html' template with the prediction result and input values
        return render_template('result.html',
                               prediction_result=result_text,
                               specific_gravity=specific_gravity,
                               hypertension=hypertension,
                               haemoglobin=haemoglobin,
                               diabetes_mellitus=diabetes_mellitus,
                               albumin=albumin,
                               serum_creatinine=serum_creatinine,
                               aanemia=aanemia,
                               pus_cell=pus_cell)
    
    except Exception as e:
        error_message = "An error occurred: " + str(e)
        return render_template('error.html', error_message=error_message)

def generate_pdf():
    pdf = canvas.Canvas("CKD_Prediction_Report.pdf")
    
    # CKD Ranges
    pdf.drawString(100, 750, "CKD Ranges:")
    pdf.drawString(100, 730, "Specific Gravity: 1.005 to 1.030")
    pdf.drawString(100, 710, "Hypertension: 90mm Hg to 140mm Hg")
    pdf.drawString(100, 690, "Hemoglobin: 12.0 to 15.5 g/dL")
    pdf.drawString(100, 670, "Diabetes Mellitus: above 126mg/dL")
    pdf.drawString(100, 650, "Albumin: above 30mg/dL")
    pdf.drawString(100, 630, "Serum Creatinine: above 5.0 mg/dL (adults)")
    pdf.drawString(100, 610, "Anemia: below 60mg/dL")
    pdf.drawString(100, 590, "Pus Cell: above 8 pus cells")

    # Prediction Result
    prediction_result = request.args.get('prediction_result', 'None')
    pdf.drawString(100, 570, "Prediction Result:")
    pdf.drawString(100, 550, f"Result: {prediction_result}")

    pdf.save()

@app.route('/generate_report_pdf', methods=['GET'])
def generate_report_pdf_route():
    try:
        generate_pdf()
        return send_file("CKD_Prediction_Report.pdf", as_attachment=True)
    except Exception as e:
        error_message = "An error occurred while generating the PDF report: " + str(e)
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
