import pandas as pd
import numpy as np
from config import dataset_path

def load_and_preprocess_data(test_fraction=0.2):
    data = pd.read_csv(dataset_path).sample(frac=1).to_numpy()  # Shuffle and convert to numpy
    m, n = data.shape
    print(f"Data: {m} x {n} matrix")

    val_set_size = int(m * test_fraction)

    # Test set
    data_test = data[:val_set_size].T
    Y_test = data_test[0]
    X_test = data_test[1:n] / 255.0

    # Training set
    data_train = data[val_set_size:].T
    Y_train = data_train[0]
    X_train = data_train[1:n] / 255.0

    return X_train, Y_train, X_test, Y_test, val_set_size