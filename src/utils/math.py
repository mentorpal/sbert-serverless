import numpy as np

def cos_sim_weight(a,b):
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)

    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))