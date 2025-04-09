import numpy as np

def uniform(arr:np.ndarray):
    norm = np.linalg.norm(arr)
    if norm != 0:
        return arr / np.linalg.norm(arr)
    return np.zeros_like(arr)

def rgba_pixel(color: str = "#ffffff", alpha: float = 1):
    carr = [int(color[i : i + 2], 16) for i in (1, 3, 5)] + [alpha * 255]
    return np.array([[carr]], np.uint8)
