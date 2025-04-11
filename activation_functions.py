import numpy as np

def jump_function(u : float) -> float:
    return 1 if u >= 0 else 0

def sigmoid_function(u: float) -> float:
    return 1 / (1 + np.exp(-u))

def relu_function(u: float) -> float:
    return u if u > 0 else 0