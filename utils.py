import numpy as np

def uniform(arr:np.ndarray):
    norm = np.linalg.norm(arr)
    if norm != 0:
        return arr / np.linalg.norm(arr)
    return np.zeros_like(arr)