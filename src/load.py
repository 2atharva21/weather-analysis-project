import joblib

def load_model():
    model = joblib.load("models/weather_model.pkl")
    print("Model loaded successfully!")

if __name__ == "__main__":
    try:
        load_model()
    except Exception as e:
        print(f"Error during model loading: {e}")
