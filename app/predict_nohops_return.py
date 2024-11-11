import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime
import os
import re

from models.sets import cyclical_transform


def encode_cyclical_features(df):
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df = df.drop(columns=['year', 'month', 'day', 'hour', 'minute'])  # Drop raw datetime features after encoding
    return df

def predict_nohops_return_flight_fare(input_date, input_time, starting_airport, destination_airport, cabin_type):
    """
    Load the prediction model and make a fare prediction.

    Parameters:
        input_date (str): The departure date in 'YYYY-MM-DD' format.
        input_time (str): The departure time in 'HH:MM' format.
        starting_airport (str): The starting airport code.
        destination_airport (str): The destination airport code.
        cabin_type (str): The cabin type (e.g., 'economy', 'business').

    Returns:
        float: Predicted fare.
    """
    
    try:
        xgb_pipe = load('models/alex_xgboost_hyperopt_new.joblib')
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the model: {e}")

    # Combine date and time into a single string
    datetime_string = f"{input_date} {input_time}"

    # Convert the string to a pandas datetime object
    departure_datetime = pd.to_datetime(datetime_string, format="%Y-%m-%d %H:%M")

    # Extract the necessary features
    input_data = {
        'startingAirport': re.search(r'\((.*?)\)', starting_airport).group(1),
        'destinationAirport': re.search(r'\((.*?)\)', destination_airport).group(1),
        'day': departure_datetime.day_name(),  
        'month': departure_datetime.month,            
        'hour': departure_datetime.hour, 
        'year': "2024",
        'minute': departure_datetime.minute,          
        'cabin_type': cabin_type
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make predictions using the loaded model
    prediction = xgb_pipe.predict(input_df)

    return prediction[0]