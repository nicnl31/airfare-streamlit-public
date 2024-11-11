import datetime
import json
from joblib import load

import pandas as pd
import numpy as np
from tensorflow import keras

from models.sets import cyclical

###############################################################################
#
#   LOAD METADATA AND MODELS
#
###############################################################################

with open('models/distance_data.json') as dist:
    distance_data = json.load(dist)

with open('models/travel_duration_data.json') as duration:
    duration_data = json.load(duration)



###############################################################################
#
#   PREDICT FUNCTION
#
###############################################################################

def predict_neural_network(
        origin: str, 
        dest: str, 
        search_date: datetime.date,
        depart_date: datetime.date, 
        depart_time: datetime.time, 
        is_basic_econ: bool,
        n_hops: int,
        cabins: str) -> float:
    try:
        model = keras.models.load_model("models/nicholas_neuralnetwork_best.keras")
        mlbCabinCode = load("models/nicholas_mlbCabinCode.joblib")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the model: {e}")
    origin_iata_code = origin[-4:-1]
    dest_iata_code = dest[-4:-1]
    cabins_series = pd.Series([cabins], name='segmentsCabinCode')
    input_data = {
        'flightDayOfWeekSin': cyclical(depart_date.weekday(), 7, np.sin), 
        'flightDayOfWeekCos': cyclical(depart_date.weekday(), 7, np.cos), 
        'flightMonthSin': cyclical(depart_date.month, 12, np.sin), 
        'flightMonthCos': cyclical(depart_date.month, 12, np.cos), 
        'flightHourSin': cyclical(depart_time.hour, 24, np.sin), 
        'flightHourCos': cyclical(depart_time.hour, 24, np.cos), 
        'flightMinuteSin': cyclical(depart_time.minute, 60, np.sin), 
        'flightMinuteCos': cyclical(depart_time.minute, 60, np.cos),
        'timeDeltaDays': (depart_date - search_date).days,
        'travelDurationDay': duration_data[origin_iata_code][dest_iata_code],
        'totalTravelDistance': distance_data[origin_iata_code][dest_iata_code],
        'isBasicEconomy': is_basic_econ,
        'isRefundable': False if is_basic_econ else True,
        'isNonStop': False if n_hops != 0 else True,
        'numLegs': n_hops,
    }
    input_df = pd.DataFrame(input_data, index=[0])
    cabins_df = pd.DataFrame(
        mlbCabinCode.transform(cabins_series), 
        columns=mlbCabinCode.classes_)
    input_df = pd.concat([input_df, cabins_df], axis=1, ignore_index=True)
    pred = model.predict(input_df)
    return pred
