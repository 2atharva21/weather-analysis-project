import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import os

def train_and_save_model():
    # Example data
    data = {
        'humidity': [0.7, 0.8, 0.6],
        'pressure': [1015, 1020, 1018],
        'wind_speed': [3.2, 3.5, 3.0],
        'temperature': [22, 24, 23]
    }
    df = pd.DataFrame(data)

    # Features and target
    X = df.drop(columns=["temperature"])
    y = df["temperature"]

    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model and scaler using pickle
    model_path = "models/weather_model.pkl"
    scaler_path = "models/scaler.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    
    with open(scaler_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    try:
        train_and_save_model()
    except Exception as e:
        print(f"Error during model training and saving: {e}")
