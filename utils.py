import numpy as np

def normalize_landmarks(landmarks):
    coords = np.array(landmarks).reshape(-1, 3).astype(np.float32)
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    norm_coords = (coords - coords_min) / (coords_max - coords_min + 1e-6)
    return norm_coords.flatten().reshape(1, -1)
