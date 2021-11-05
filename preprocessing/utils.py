import numpy as np

def normalize(data, max_factor, min_factor, max_bound, min_bound):
    norm_data = min_bound + (data - min_factor) * (max_bound - min_bound) / (max_factor - min_factor)
    return norm_data