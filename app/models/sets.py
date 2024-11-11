import os.path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='../data/processed/'):
    """Save the different sets locally

    Parameters
    ----------
    X_train: Numpy Array or DataFrame
        Features for the training set
    y_train: Numpy Array or DataFrame
        Target for the training set
    X_val: Numpy Array or DataFrame
        Features for the validation set
    y_val: Numpy Array or DataFrame
        Target for the validation set
    X_test: Numpy Array or DataFrame
        Features for the testing set
    y_test: Numpy Array or DataFrame
        Target for the testing set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')

    Returns
    -------
    """

    if X_train is not None:
        pd.DataFrame(X_train).to_parquet(f'{path}X_train.parquet', index=False)
    if X_val is not None:
        pd.DataFrame(X_val).to_parquet(f'{path}X_val.parquet', index=False)
    if X_test is not None:
        pd.DataFrame(X_test).to_parquet(f'{path}X_test.parquet', index=False)
    if y_train is not None:
        pd.Series(y_train).to_frame(name='target').to_parquet(f'{path}y_train.parquet', index=False)
    if y_val is not None:
        pd.Series(y_val).to_frame(name='target').to_parquet(f'{path}y_val.parquet', index=False)
    if y_test is not None:
        pd.Series(y_test).to_frame(name='target').to_parquet(f'{path}y_test.parquet', index=False)


def load_sets(path='../data/processed/'):
    """Load the different locally saved sets

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    pd.DataFrame
        Features for the training set
    pd.Series
        Target for the training set
    pd.DataFrame
        Features for the validation set
    pd.Series
        Target for the validation set
    pd.DataFrame
        Features for the testing set
    pd.Series
        Target for the testing set
    """
    
    X_train = pd.read_parquet(f'{path}X_train.parquet') if os.path.isfile(f'{path}X_train.parquet') else None
    X_val   = pd.read_parquet(f'{path}X_val.parquet') if os.path.isfile(f'{path}X_val.parquet') else None
    X_test  = pd.read_parquet(f'{path}X_test.parquet') if os.path.isfile(f'{path}X_test.parquet') else None
    y_train = pd.read_parquet(f'{path}y_train.parquet')['target'] if os.path.isfile(f'{path}y_train.parquet') else None
    y_val   = pd.read_parquet(f'{path}y_val.parquet')['target'] if os.path.isfile(f'{path}y_val.parquet') else None
    y_test  = pd.read_parquet(f'{path}y_test.parquet')['target'] if os.path.isfile(f'{path}y_test.parquet') else None

    return X_train, y_train, X_val, y_val, X_test, y_test


def split_sets_by_time(df, target_col, test_ratio=0.2):
    """Split sets by indexes for an ordered dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    pd.DataFrame
        Features for the training set
    pd.Series
        Target for the training set
    pd.DataFrame
        Features for the validation set
    pd.Series
        Target for the validation set
    pd.DataFrame
        Features for the testing set
    pd.Series
        Target for the testing set
    """
    
    df_copy = df.copy()
    target = df_copy.pop(target_col)
    cutoff = int(len(df_copy) * test_ratio)

    # Define training, validation, and testing sets based on indices
    X_train = df_copy.iloc[: -cutoff * 2]
    y_train = target.iloc[: -cutoff * 2]
    
    X_val = df_copy.iloc[-cutoff * 2: -cutoff]
    y_val = target.iloc[-cutoff * 2: -cutoff]
    
    X_test = df_copy.iloc[-cutoff:]
    y_test = target.iloc[-cutoff:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def cyclical_transform(X):
    """
    Transform cyclical features into sine and cosine components.

    This function takes a DataFrame containing cyclical features related to time,
    such as month, hour, and minute, and returns a new DataFrame with their
    corresponding sine and cosine transformations.

    Parameters
    ----------
    X : pd.DataFrame
        A DataFrame containing the following cyclical columns:
        - 'departure_month': Integer values representing the month (1 to 12).
        - 'departure_hour': Integer values representing the hour of the day (0 to 23).
        - 'departure_minute': Integer values representing the minute of the hour (0 to 59).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the transformed cyclical features:
        - 'month_sin': Sine transformation of the month.
        - 'month_cos': Cosine transformation of the month.
        - 'hour_sin': Sine transformation of the hour.
        - 'hour_cos': Cosine transformation of the hour.
        - 'minute_sin': Sine transformation of the minute.
        - 'minute_cos': Cosine transformation of the minute.
    """
    
    X_transformed = pd.DataFrame()
    X_transformed['month_sin'] = np.sin(2 * np.pi * X['departure_month'] / 12)
    X_transformed['month_cos'] = np.cos(2 * np.pi * X['departure_month'] / 12)
    X_transformed['hour_sin'] = np.sin(2 * np.pi * X['departure_hour'] / 24)
    X_transformed['hour_cos'] = np.cos(2 * np.pi * X['departure_hour'] / 24)
    X_transformed['minute_sin'] = np.sin(2 * np.pi * X['departure_minute'] / 60)
    X_transformed['minute_cos'] = np.cos(2 * np.pi * X['departure_minute'] / 60)
    
    return X_transformed


def cyclical(data: pd.Series or int, max_data: int or float, func=np.sin):
    """
    Transform a single cyclical feature into either sine or cosine components.
    """
    assert func in [np.sin, np.cos], "Function must be either sine or cosine."
    return func(2 * np.pi * data / max_data)
