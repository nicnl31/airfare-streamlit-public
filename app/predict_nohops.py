import pandas as pd
from joblib import load
from datetime import datetime
import os

from models.sets import cyclical_transform


def predict_nohops_flight_fare(input_date, input_time, starting_airport, destination_airport, cabin_type):
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
        xgb_pipe = load('models/pine_xgb_pipeline_final.joblib')
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
        'startingAirport': starting_airport,
        'destinationAirport': destination_airport,
        'departure_dayofweek': departure_datetime.day_name(),  
        'departure_month': departure_datetime.month,            
        'departure_hour': departure_datetime.hour,              
        'departure_minute': departure_datetime.minute,          
        'cabin_type': cabin_type
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make predictions using the loaded model
    prediction = xgb_pipe.predict(input_df)

    return prediction[0]
