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
        result_text = "CKD" if prediction_result == 1 else "Not CKD"
        
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
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # CKD Ranges
    pdf.cell(200, 10, txt="CKD Ranges:", ln=True, align='L')
    pdf.multi_cell(0, 10, "Specific Gravity: 1.005 to 1.030\n"
                           "Hypertension: 90mm Hg to 140mm Hg\n"
                           "Hemoglobin: 12.0 to 15.5 g/dL\n"
                           "Diabetes Mellitus: above 126mg/dL\n"
                           "Albumin: above 30mg/dL\n"
                           "Serum Creatinine: above 5.0 mg/dL (adults)\n"
                           "Anemia: below 60mg/dL\n"
                           "Pus Cell: above 8 pus cells", align='L')

    # Prediction Result
    prediction_result = request.args.get('prediction_result', 'None')
    pdf.cell(200, 10, txt="\nPrediction Result:", ln=True, align='L')
    pdf.multi_cell(0, 10, f"Result: {prediction_result}", align='L')

    # Output the PDF content to a temporary buffer
    pdf_buffer = pdf.output(dest='S')

    # Specify the PDF file path
    pdf_file = "CKD_Prediction_Report.pdf"

    # Write the PDF content from the buffer to the file
    with open(pdf_file, 'wb') as f:
        f.write(pdf_buffer)

    return pdf_file

@app.route('/generate_report_pdf_route', methods=['GET'])
def generate_report_pdf_route():
    try:
        pdf_file = generate_pdf()
        def generate():
            with open(pdf_file, 'rb') as f:
                pdf_data = f.read()
                yield pdf_data
        response = Response(generate(), content_type='application/pdf')
        response.headers['Content-Disposition'] = 'attachment; filename=CKD_Prediction_Report.pdf'
        return response
    

    except Exception as e:
        error_message = "An error occurred while generating the PDF report: " + str(e)
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
