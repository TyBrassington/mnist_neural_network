import numpy as np
import pandas as pd
from config import *


# Initialize Parameters

def init_params():
    params = {
        "W1": np.random.randn(HIDDEN_LAYER_1_SIZE, INPUT_SIZE) * 0.01,
        "b1": np.zeros((HIDDEN_LAYER_1_SIZE, 1)),
        "W2": np.random.randn(HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_1_SIZE) * 0.01,
        "b2": np.zeros((HIDDEN_LAYER_2_SIZE, 1)),
        "W3": np.random.randn(OUTPUT_CLASSES, HIDDEN_LAYER_2_SIZE) * 0.01,
        "b3": np.zeros((OUTPUT_CLASSES, 1)),
    }
    return params

# Model Saving and Loading
def save_params(params, filepath, current_iteration=0, autosave_state=False):
    # Convert each parameter to DataFrame and save as CSV
    for key, value in params.items():
        pd.DataFrame(value).to_csv(f"{filepath}/{key}.csv", index=False)
    if autosave_state:
        pd.DataFrame({"current_iteration": [current_iteration]}).to_csv(f"{filepath}/state.csv", index=False)
        print(f"Training state saved to {filepath}")
    else:
        print(f"Weights and biases saved to {filepath}")

def load_params(filepath, autosave_state=False):
    # Read each CSV file into a DataFrame and convert to numpy array
    params = {}
    for param_name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        params[param_name] = pd.read_csv(f"{filepath}/{param_name}.csv").values
    if autosave_state:
        current_iteration = int(pd.read_csv(f"{filepath}/state.csv")["current_iteration"][0])
        print(f"Training state loaded from {filepath}")
    else:
        current_iteration = 0
        print(f"Weights and biases loaded from {filepath}")
    return params, current_iteration