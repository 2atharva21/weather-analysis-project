import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(filename):
    # Read the data
    df = pd.read_csv(filename)
    
    # Check if required columns exist
    required_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'description']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Fill missing values with forward fill
    df.fillna(method='ffill', inplace=True)
    
    # One-hot encode the 'description' column
    df = pd.get_dummies(df, columns=["description"], drop_first=True)
    
    # Normalize numerical columns
    scaler = MinMaxScaler()
    df[['temperature', 'humidity', 'pressure', 'wind_speed']] = scaler.fit_transform(df[['temperature', 'humidity', 'pressure', 'wind_speed']])
    
    return df

if __name__ == "__main__":
    try:
        # Preprocess data and save to a new CSV
        processed_df = preprocess_data("data/raw_data.csv")
        processed_df.to_csv("data/processed_data.csv", index=False)
        print("Data preprocessing completed!")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
