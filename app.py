from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('earthquake_model.pkl')
region_encoder = joblib.load('region_encoder.pkl')
soil_encoder = joblib.load('soil_encoder.pkl')
risk_encoder = joblib.load('risk_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    depth = float(request.form['depth'])
    region = request.form['region']
    tremors = int(request.form['tremors'])
    soil = request.form['soil']

    # Encode categorical data
    region_encoded = region_encoder.transform([region])[0]
    soil_encoded = soil_encoder.transform([soil])[0]

    # Prepare input
    input_data = np.array([[latitude, longitude, depth,
                            region_encoded, tremors, soil_encoded]])

    # Prediction
    prediction = model.predict(input_data)

    result = risk_encoder.inverse_transform(prediction)[0]

    return render_template('index.html', prediction_text=
    f"Earthquake Risk Level: {result}")

if __name__ == '__main__':
    app.run(debug=True)