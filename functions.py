import numpy as np
from math import *
import scipy
def wood(X):
    """[wood function]
    Args:
        X ([np.array]): Input X

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
        (sum(((X[idx] + 10 * X[idx + 1])**2,
                5 * (X[idx+2] - X[idx+3])**2,
                (X[idx+1] - X[idx+2])**4,
                10 * (X[idx] - X[idx+3])**4,
            )) for idx in range(0, len(X), 4)))

def trigonometric(X):
    n = len(X)
    sum_cos = sum((cos(x) for x in X))
    return sum(
        ( (n - sum_cos + (idx + 1) * (1 - cos(x)) - sin(x)) ** 2 for idx, x in enumerate(X))
    )

def test():
    print("wood funtion")
    print(wood([1,1,1,1]))
    print("extended_powell_singular")
    for i in [20, 40, 60, 80 ,100]:
        print(extended_powell_singular([0]*i))
    print("trigonometric")
    for i in [20, 40, 60, 80 ,100]:
        print(trigonometric([0 for _ in range(i)]))