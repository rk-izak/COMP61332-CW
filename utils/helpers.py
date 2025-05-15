import os
import joblib
import pandas as pd


def readData(data_path):
    """
    Read data from a JSON file and return a pandas DataFrame.

    Args:
        data_path (str): The path to the JSON file.

    Returns:
        pd.DataFrame: The data read from the JSON file.

    Raises:
        Exception: If the data file is not found or an error occurs while reading the file.
    """
    if not os.path.exists(data_path):
        raise Exception("No data found at given path.")

    try:
        df = pd.read_json(data_path)
    except Exception as e:
        raise Exception(f"Error reading data file: {e}")

    return df


def save_model(model, name='unnamed'):
    """
    Save a trained model to a file using joblib.

    Args:
        model: The trained model object.
        name (str, optional): The name of the model. Defaults to 'unnamed'.

    Returns:
        None
    """
    os.makedirs('checkpoints', exist_ok=True)
    joblib.dump(model, f'checkpoints/{name}-model.joblib')