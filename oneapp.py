from flask import Flask, render_template, request, Response, send_file
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

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
        result_text = "CKD Positive" if prediction_result == 1 else " CKD Negative"
        
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
    # Create a PDF document
    doc = SimpleDocTemplate("CKD_Prediction_Report.pdf")
    elements = []

    # Add hospital name in bold with color
    hospital_name = "<font name='Helvetica-Bold' size='16' color='#FF0000'>Your Hospital Name</font>"
    elements.append(hospital_name)

    # CKD Ranges
    ranges_data = [
        ["CKD Ranges:"],
        ["Specific Gravity", "1.005 to 1.030"],
        ["Hypertension (Blood Pressure)", "90 mm Hg to 140 mm Hg"],
        ["Haemoglobin", "12.0 to 15.5 g/dL"],
        ["Diabetes Mellitus (Blood Sugar)", "Above 126 mg/dL"],
        ["Albumin", "Above 30 mg/dL"],
        ["Serum Creatinine", "Above 5.0 mg/dL (adults)"],
        ["Anemia", "Below 60 mg/dL"],
        ["Pus Cell (Urine Analysis)", "Above 8 pus cells"]
    ]

    # Create a table for CKD Ranges and format it
    ranges_table = Table(ranges_data)
    ranges_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    elements.append(ranges_table)

    # Prediction Result
    prediction_result = request.args.get('prediction_result', 'None')
    result_data = [
        ["Prediction Result:"],
        ["Result", prediction_result]
    ]

    # Create a table for Prediction Result and format it
    result_table = Table(result_data)
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    elements.append(result_table)

    # Build the PDF document
    doc.build(elements)
    pdf = canvas.Canvas("CKD_Prediction_Report.pdf")

    # Print a debug message to check if this function is called
    print("Generating PDF...")

    # CKD Ranges
    pdf.drawString(100, 750, "CKD Ranges:")
    # Add more drawing commands here...

    # Save the PDF
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
