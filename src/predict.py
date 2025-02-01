import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def train_and_save_model():
    # Example training data (use real data here)
    X_train = [[0.7, 1015, 3.2], [0.6, 1020, 4.0], [0.8, 1018, 3.5]]
    y_train = [22.3, 23.0, 21.7]
    
    # Train the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, "models/weather_model.pkl")
    
    # Save scaler if you're using it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, "models/scaler.pkl")

def predict_weather(input_data):
    # Load the trained model (ensure it’s the most recent model)
    model = joblib.load("models/weather_model.pkl")
    
    # Load the scaler used for normalization (during training)
    scaler = joblib.load("models/scaler.pkl")
    
    # List of expected features (based on the model training process)
    expected_columns = ['humidity', 'pressure', 'wind_speed']
    
    # Convert the input_data into a DataFrame
    df = pd.DataFrame([input_data])
    
    # Ensure the input data matches the expected columns
    df = df.reindex(columns=expected_columns, fill_value=0)
    
    # Normalize the numerical features using the same scaler as during training
    df[['humidity', 'pressure', 'wind_speed']] = scaler.transform(df[['humidity', 'pressure', 'wind_speed']])
    
    # Make prediction without feature names
    prediction = model.predict(df.values)  # Use .values to drop feature names
    
    return prediction[0]

if __name__ == "__main__":
    # Train model (this step only needs to be done once and will save the model)
    # train_and_save_model()  # Uncomment this to retrain and save model
    
    # Example input data for prediction (this would ideally be dynamic)
    input_data = {
        'humidity': float(input("Enter humidity: ")), 
        'pressure': float(input("Enter pressure: ")), 
        'wind_speed': float(input("Enter wind speed: "))
    }
    
    # Make prediction each time with updated input data
    temp = predict_weather(input_data)
    print(f"Predicted Temperature: {temp:.2f}°C")
