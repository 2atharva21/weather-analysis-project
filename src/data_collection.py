import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import time

load_dotenv()

BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Set OPENWEATHER_API_KEY environment variable.")

# List of cities to fetch data for
cities = ["Pune", "Mumbai", "Delhi", "Bangalore", "Chennai"]

def fetch_weather(city):
    params = {
        'q': city,
        'appid': API_KEY,
        'units': 'metric'
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('weather'):
            raise ValueError("No weather information available.")
        
        weather = {
            "city": city,
            "temperature": data.get("main", {}).get("temp"),
            "humidity": data.get("main", {}).get("humidity"),
            "pressure": data.get("main", {}).get("pressure"),
            "wind_speed": data.get("wind", {}).get("speed"),
            "description": data["weather"][0].get("description"),
            "timestamp": datetime.now().isoformat()
        }
        print(weather)  # Check the fetched weather data
        return weather
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except (KeyError, IndexError, ValueError) as e:
        print(f"Data parsing error: {e}")
    return None

def collect_weather_data():
    weather_data = []
    for city in cities:
        print(f"Fetching data for {city}...")
        weather = fetch_weather(city)
        if weather:
            weather_data.append(weather)
        time.sleep(1)  # Sleep to avoid hitting the API rate limit
    return weather_data

if __name__ == "__main__":
    weather_data = collect_weather_data()

    if weather_data:
        os.makedirs("../data", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/raw_data.csv"
        
        # Save the collected weather data to CSV
        df = pd.DataFrame(weather_data)
        df.to_csv(filename, index=False)
        print(f"Weather data saved to {filename}")
else:
        print("No weather data collected.")