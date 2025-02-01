from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("../models/weather_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Convert the incoming data into a DataFrame
    df = pd.DataFrame([data])
    
    # List of expected features (based on the model training process)
    expected_columns = ['humidity', 'pressure', 'wind_speed']
    
    # Ensure the input data matches the expected columns
    df = df.reindex(columns=expected_columns, fill_value=0)
    
    # Normalize the numerical features using the same scaler as during training
    df[['humidity', 'pressure', 'wind_speed']] = scaler.transform(df[['humidity', 'pressure', 'wind_speed']])
    
    # Make prediction
    prediction = model.predict(df)
    
    # Return the predicted temperature
    return jsonify({"predicted_temperature": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
