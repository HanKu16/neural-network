def jump_function(u : float) -> float:
    return 1 if u >= 0 else 0

def sigmoid_function(u: float) -> float:
    e : float = 2.718281828459045
    return 1 / (1 + e ** (-u))

def relu_function(u: float) -> float:
    return u if u > 0 else 0