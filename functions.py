import numpy as np
import 
def wood(X):
    """[wood function]
    Args:
        X ([list or np.array]): Input X

    Returns:
        [float]: funciton values
    """    
    x1 = X[0]
    x2 = X[1]
    x3 = X[2]
    x4 = X[3]
    return sum((
        100 * (x1 * x1 - x2)**2,
        (x1 - 1)**2,
        (x3 - 1)**2,
        90 * (x3 * x3 - x4)**2,
        10.1 * ((x2 - 1)**2 + (x4 - 1)**2),
        19.8 * (x2 - 1) * (x4 - 1),
    ))


def extended_powell_singular(X):
    assert len(X) % 4 == 0, "Len of X must be a multiple of 4"
    return sum(
        (sum(((X[idx] + 10 * X[idx + 1])**2)) for idx in range(0, len(X), 4)))
