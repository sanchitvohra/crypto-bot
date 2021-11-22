import numpy as np

def load_data():
    data = np.load("data/crypto_data.npy")
    return data