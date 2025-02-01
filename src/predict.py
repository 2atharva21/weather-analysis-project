import joblib
import pandas as pd

def predict_weather(input_data):
    # Load the trained model and scaler
    model = joblib.load("models/weather_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    
    # List of all features that the model expects, excluding the target variable
    expected_columns = ['humidity', 'pressure', 'wind_speed', 'description_clear sky', 'description_cloudy']  # Adjust based on your dataset
    
    # Convert the input_data into a DataFrame
    df = pd.DataFrame([input_data])
    
    # If description column exists in input data, one-hot encode it
    if 'description' in input_data:
        description_dummies = pd.get_dummies(df['description'], drop_first=True)
        df = pd.concat([df, description_dummies], axis=1).drop(columns=['description'])
    
    # Align the input columns to match the model's expected features
    df = df.reindex(columns=expected_columns, fill_value=0)
    
    # Normalize the numerical features (use the same scaler as during training)
    df[['humidity', 'pressure', 'wind_speed']] = scaler.transform(df[['humidity', 'pressure', 'wind_speed']])
    
    # Make prediction
    prediction = model.predict(df)
    
    return prediction[0]

if __name__ == "__main__":
    input_data = {
        'humidity': 0.7, 
        'pressure': 1015, 
        'wind_speed': 3.2,
        'description': 'clear sky'  # Adjust this to match the model's features
    }
    
    temp = predict_weather(input_data)
    print(f"Predicted Temperature: {temp:.2f}°C")
