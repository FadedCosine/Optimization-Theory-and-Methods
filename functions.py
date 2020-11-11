import numpy as np
import math
import scipy
import sympy
from sympy import diff
from sympy import symbols
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

def diff_wood():
    """[wood function的导函数]
    """
    x1, x2, x3, x4 = symbols("x1,x2,x3,x4")
    wood_func = 100 * (x1 ** 2 - x2)**2 + (x1 - 1)**2 + (x3 - 1)**2 + 90 * (x3 ** 2 - x4)**2 + 10.1 * ((x2 - 1)**2 + (x4 - 1)**2) + 19.8 * (x2 - 1) * (x4 - 1)
    diff_x1 = diff(wood_func, x1)
    diff_x2 = diff(wood_func, x2)
    diff_x3 = diff(wood_func, x3)
    diff_x4 = diff(wood_func, x4)
    # return [diif_x1.subs([(x1, X[0]), (x2, X[1]), (x3, X[2]), (x4, X[3])]), diif_x2.subs([(x1, X[0]), (x2, X[1]), (x3, X[2]), (x4, X[3])]), diif_x3.subs([(x1, X[0]), (x2, X[1]), (x3, X[2]), (x4, X[3])]), diif_x4.subs([(x1, X[0]), (x2, X[1]), (x3, X[2]), (x4, X[3])])]
    return [diff_x1, diff_x2, diff_x3, diff_x4], (x1, x2, x3, x4)

def g_wood(X, diff_list=None, symbols_list=None):
    """[计算wood函数在X出的一阶导数值]
    Args:
        X ([np.array]): Input X
        diff_list ([list]): 导函数分量列表
        symbols_list ([list]): 导函数的变量符号列表
    Returns:
        [float]: wood函数在X出的一阶导数值
    """
    if diff_list is not None:
        return np.array([diff_xi.subs([(symbol, x_i) for symbol, x_i in zip(symbols_list, X)]) for diff_xi in diff_list])


def extended_powell_singular(X):
    assert len(X) % 4 == 0, "Len of X must be a multiple of 4"
    return sum(
        (sum(((X[idx] + 10 * X[idx + 1])**2,
                5 * (X[idx+2] - X[idx+3])**2,
                (X[idx+1] - X[idx+2])**4,
                10 * (X[idx] - X[idx+3])**4,
            )) for idx in range(0, len(X), 4)))

def diff_extended_powell_singular(m):
    """[extended_powell_singular 函数的导函数]
    Args:
        m ([int]): X的维度
    """
    assert m % 4 == 0, "Len of X must be a multiple of 4"
    symbols_X = symbols("x:{}".format(m))
    eps_func = 0
    for idx in range(0, m, 4):
        eps_func += (symbols_X[idx] + 10 * symbols_X[idx + 1])**2 + \
                5 * (symbols_X[idx+2] - symbols_X[idx+3])**2 + \
                (symbols_X[idx+1] - symbols_X[idx+2])**4 + \
                10 * (symbols_X[idx] - symbols_X[idx+3])**4
    
    diff_list = []
    for symbol in symbols_X:
        diff_list.append(diff(eps_func, symbol))
    return diff_list, symbols_X

def trigonometric(X):
    n = len(X)
    sum_cos = sum((math.cos(x) for x in X))
    return sum(
        ( (n - sum_cos + (idx + 1) * (1 - math.cos(x)) - math.sin(x)) ** 2 for idx, x in enumerate(X))
    )

def diff_trigonometric(m):
    """[trigonometric 函数的导函数]
    Args:
        m ([int]): X的维度
    """
    symbols_X = symbols("x:{}".format(m))
    sum_cos = sum((sympy.cos(x) for x in symbols_X))
    trig_func = sum(
        ( (m - sum_cos + (idx + 1) * (1 - sympy.cos(x)) - sympy.sin(x)) ** 2 for idx, x in enumerate(symbols_X))
    )
    diff_list = []
    for symbol in symbols_X:
        diff_list.append(diff(trig_func, symbol))
    return diff_list, symbols_X
 

def test():
    
    x0 = np.array([-3, -1, -3, -1])
    diff_wood_list, symbols_wood_list = diff_wood()
    print(g_wood(x0, diff_wood_list, symbols_wood_list))
    # for m in [20, 40, 60 ,80, 100]:
    #     x0 = np.array([0] * m)
    #     diff_eps_list, symbols_eps_list = diff_extended_powell_singular(m)
    #     print([diff_xi.subs([(symbol, x_i) for symbol, x_i in zip(symbols_eps_list, x0)]) for diff_xi in diff_eps_list])
    # for m in [20, 40, 60, 80 ,100]:
    #     x0 = np.array([0] * m)
    #     diff_trig_list, symbols_trig_list = diff_trigonometric(m)
    #     print([diff_xi.subs([(symbol, x_i) for symbol, x_i in zip(symbols_trig_list, x0)]) for diff_xi in diff_trig_list])

def main():
    test()

if __name__ == "__main__":
    main()