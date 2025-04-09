from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
with open('loan_model_4.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    applicant_income = float(request.form['applicant_income'])
    coapplicant_income = float(request.form['coapplicant_income'])
    loan_amount = float(request.form['loan_amount'])
    loan_amount_term = float(request.form['loan_amount_term'])
    credit_history = 1 if request.form['credit_history'] == 'No' else 0
    gender = 1 if request.form['gender'] == 'Male' else 0
    married = 1 if request.form['married'] == 'Yes' else 0
    education = 1 if request.form['education'] == 'Graduate' else 0
    self_employed = 1 if request.form['self_employed'] == 'Yes' else 0
    dependents = int(request.form['dependents'])

    property_area = request.form['property_area']
    if property_area == 'Urban':
        property_area = 2
    elif property_area == 'Semiurban':
        property_area = 1
    else:
        property_area = 0

    input_features = np.array([[applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                                credit_history, gender, married, dependents, education,
                                self_employed, property_area]])



    # Predict
    prediction = model.predict(input_features)

    result = 'Approved' if prediction[0] == 1 else 'Rejected'
    return render_template('index.html', prediction_text=f'Loan Status: {result}')

if __name__ == '__main__':
    app.run(debug=True)  
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
