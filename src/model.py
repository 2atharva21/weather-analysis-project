from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd
import os

def train_model():
    processed_data_path = "data/processed_data.csv"
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"{processed_data_path} does not exist.")
    
    df = pd.read_csv(processed_data_path)
    
    # Assuming the data is already cleaned and preprocessed
    X = df.drop(columns=["temperature", "city", "timestamp"])  # Adjust columns as per your data
    y = df["temperature"]
    
    # Normalize numerical columns
    scaler = MinMaxScaler()
    X[['humidity', 'pressure', 'wind_speed']] = scaler.fit_transform(X[['humidity', 'pressure', 'wind_speed']])
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model and scaler
    model_path = "../models/weather_model.pkl"
    scaler_path = "../models/scaler.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model and the scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error during model training: {e}")
