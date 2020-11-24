import numpy as np
import math
import scipy
import sympy
from sympy import diff
from sympy import symbols
import functools
import pickle 
from numpy import sin, cos, sum
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

def diff_wood_expression():
    """[wood function的导函数的表达式]
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
    """[计算wood函数在X处的一阶导数值]
    Args:
        X ([np.array]): Input X
        diff_list ([list]): 导函数分量列表
        symbols_list ([list]): 函数的变量符号列表
    Returns:
        [float]: wood函数在X出的一阶导数值
    """
    if diff_list is not None:
        return np.array([diff_xi.subs([(symbol, x_i) for symbol, x_i in zip(symbols_list, X)]) for diff_xi in diff_list], 'float')

def hess_wood_expression():
    G = [[] for _ in range(4)]

    x1, x2, x3, x4 = sympy.symbols('x1,x2,x3,x4')
    wood_func = 100 * (x1 ** 2 - x2)**2 + (x1 - 1)**2 + (x3 - 1)**2 + 90 * (x3 ** 2 - x4)**2 + 10.1 * ((x2 - 1)**2 + (x4 - 1)**2) + 19.8 * (x2 - 1) * (x4 - 1)
    
    gx_1 = sympy.diff(wood_func, x1)
    gx_2 = sympy.diff(wood_func, x2)
    gx_3 = sympy.diff(wood_func, x3)
    gx_4 = sympy.diff(wood_func, x4)

    Gx_11 = sympy.diff(gx_1, x1)
    Gx_12 = sympy.diff(gx_1, x2)
    Gx_13 = sympy.diff(gx_1, x3)
    Gx_14 = sympy.diff(gx_1, x4)

    Gx_22 = sympy.diff(gx_2, x2)
    Gx_23 = sympy.diff(gx_2, x3)
    Gx_24 = sympy.diff(gx_2, x4)

    Gx_33 = sympy.diff(gx_3, x3)
    Gx_34 = sympy.diff(gx_3, x4)

    Gx_44 = sympy.diff(gx_4, x4)

    G[0].extend([Gx_11, Gx_12, Gx_13, Gx_14])
    G[1].extend([Gx_12, Gx_22, Gx_23, Gx_24])
    G[2].extend([Gx_13, Gx_23, Gx_33, Gx_34])
    G[3].extend([Gx_14, Gx_24, Gx_34, Gx_44])

    return G, (x1, x2, x3, x4)

def G_wood(X, G_lists=None, symbols_list=None):
    """[计算wood函数在X处的Hess矩阵值]
    Args:
        X ([np.array]): Input X
        G_list ([list]): hess矩阵表达式分量二维列表
        symbols_list ([list]): 函数的变量符号列表
    Returns:
        [float]: wood函数在X出的一阶导数值
    """
    if G_lists is not None:
        return np.array([[G_xi.subs([(symbol, x_i) for symbol, x_i in zip(symbols_list, X)]) for G_xi in G_list] for G_list in G_lists], 'float')

def extended_powell_singular(X):
    assert len(X) % 4 == 0, "Len of X must be a multiple of 4"
    return sum(
        (sum(((X[idx] + 10 * X[idx + 1])**2,
                5 * (X[idx+2] - X[idx+3])**2,
                (X[idx+1] - 2 * X[idx+2])**4,
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
                (symbols_X[idx+1] - 2 * symbols_X[idx+2])**4 + \
                10 * (symbols_X[idx] - symbols_X[idx+3])**4
    
    diff_list = []
    for symbol in symbols_X:
        diff_list.append(diff(eps_func, symbol))
    return diff_list, symbols_X
def g_EPS(X):
    """ 输入X,手动计算EPS的一阶导数
    Args:
        X ([np.array]): Input X
    Returns:
        [[np.array]]: 输出在X处的
    """
    m = len(X)
    assert m % 4 == 0  # m should be exactly divisible by 4
    g = np.zeros(m)
    for iter in range(0, int(m / 4)):
        g[4 * iter] = 2 * (X[4 * iter] + 10 * X[4 * iter + 1]) + 40 * math.pow(
            X[4 * iter] - X[4 * iter + 3], 3)
        g[4 * iter + 1] = 20 * (X[4 * iter] + 10 * X[4 * iter + 1]) + 4 * math.pow(
            X[4 * iter + 1] - 2 * X[4 * iter + 2], 3)
        g[4 * iter + 2] = 10 * (X[4 * iter + 2] - X[4 * iter + 3]) - 8 * math.pow(X[4 * iter + 1] - 2 * X[4 * iter + 2],
                                                                                  3)
        g[4 * iter + 3] = -10 * (X[4 * iter + 2] - X[4 * iter + 3]) - 40 * math.pow(X[4 * iter] - X[4 * iter + 3],
                                                                                    3)
    return g

def G_EPS(X):
    """ 输入X,手动计算EPS的hess矩阵
    Args:
        X ([np.array]): Input X
    Returns:
        [[np.array]]: 输出在X处的hess矩阵值
    """
    m = len(X)
    assert m % 4 == 0  # m should be exactly divisible by 4
    G = np.zeros((m, m))
    for iter in range(0, int(m / 4)):
        x1 = X[4 * iter]
        x2 = X[4 * iter + 1]
        x3 = X[4 * iter + 2]
        x4 = X[4 * iter + 3]

        G[4 * iter][4 * iter] = 2 + 120 * (x1 - x4) ** 2
        G[4 * iter][4 * iter + 1] = 20
        G[4 * iter][4 * iter + 3] = -120 * (x1 - x4) ** 2

        G[4 * iter + 1][4 * iter] = 20
        G[4 * iter + 1][4 * iter + 1] = 200 + 12 * (x2 - 2 * x3) ** 2
        G[4 * iter + 1][4 * iter + 2] = -24 * (x2 - 2 * x3) ** 2
        
        G[4 * iter + 2][4 * iter + 1] = G[4 * iter + 1][4 * iter + 2]
        G[4 * iter + 2][4 * iter + 2] = 10 + 48 * (x2 - 2 * x3) ** 2
        G[4 * iter + 2][4 * iter + 3] = -10
        
        G[4 * iter + 3][4 * iter] = G[4 * iter][4 * iter + 3]
        G[4 * iter + 3][4 * iter + 2] = G[4 * iter + 2][4 * iter + 3]
        G[4 * iter + 3][4 * iter + 3] = 10 + 120 * (x1 - x4) ** 2
    return G


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
 
def g_trigonometric(x):
    """ 输入X,手动计算trigonometric的一阶导数
    Args:
        X ([np.array]): Input X
    Returns:
        [[np.array]]: 输出在X处的一阶导数值
    """
    n = len(x)
    X = x.reshape(-1,1)
    one = np.array([i + 1 for i in range(n)]).reshape(-1, 1)
    constant = n - sum(cos(X))
    gamma = constant + one * (1 - cos(X)) - sin(X)
    gamma_sum = sum(constant + one * (1 - cos(X)) - sin(X))
    g = 2 * gamma * (one * sin(X) - cos(X)) + 2 * sin(X) * gamma_sum
    return g.reshape(n)

def G_trigonometric(x):
    """ 输入X,手动计算trigonometric的hess矩阵
    Args:
        X ([np.array]): Input X
    Returns:
        [[np.array]]: 输出在X处的hess矩阵值
    """
    n = len(x)
    X = x.reshape(-1,1)
    constant = n - sum(cos(X))
    one = np.array([i + 1 for i in range(n)]).reshape(-1, 1)
    gamma_sum = sum(constant + one * (1 - cos(X)) - sin(X))
    diag = 2 * (sin(X) + one * sin(X) - cos(X)) * (one * sin(X) -cos(X)) + 2 * (constant + one * (1 - cos(X)) - sin(X)) * \
                            (one * cos(X) + sin(X)) + 2 * cos(X) * gamma_sum + 2 * sin(X) * \
                            (n * sin(X) + one * sin(X) - cos(X))
    diag = diag.reshape(-1)
    G = 2 * np.matmul((one * sin(X) - cos(X)), sin(X).T) + 2 * np.matmul(sin(X), (n * sin(X) + one * sin(X) - cos(X)).T)
    for i in range(n):
        G[i][i] = diag[i]
    return G


def g_function(X, diff_list=None, symbols_list=None):
    """ 输入一阶导数的符号表达式，计算一阶导数值
    Args:
        X ([np.array]): Input X
        diff_list ([list]): 导函数表达式分量列表
        symbols_list ([list]): 函数的变量符号列表
    Returns:
        [float]: 输出在X处的一阶导数值
    """
    if diff_list is not None:
        return np.array([diff_xi.subs([(symbol, x_i) for symbol, x_i in zip(symbols_list, X)]) for diff_xi in diff_list], 'float')

def hess_expression(m, diff_list=None, symbols_list=None):
    """ 输入一阶导数的符号表达式，计算hess矩阵的符号表达式
    Args:
        X ([np.array]): Input X
        diff_list ([list]): 导函数表达式分量列表
        symbols_list ([list]): 函数的变量符号列表
    Returns:
        hess矩阵的符号表达式
    """
    G = [[] for _ in range(m)]
    for i in range(m):
        for j in range(i+1):
            G[i].append(sympy.diff(diff_list[i], symbols_list[j]))
    for j in range(m):
        for i in range(j+1,m):
            G[j].append(G[i][j])
    return G, symbols_list

def G_function(X, G_lists=None, symbols_list=None):
    """ 输入hess矩阵的符号表达式，计算hess矩阵
    Args:
        X ([np.array]): Input X
        G_lists ([list]): hess矩阵的符号表达式
        symbols_list ([list]): 函数的变量符号列表
    Returns:
        [[np.array]]: 输出在X处的hess矩阵值
    """
    if G_lists is not None:
        return np.array([[G_xi.subs([(symbol, x_i) for symbol, x_i in zip(symbols_list, X)]) for G_xi in G_list] for G_list in G_lists], 'float')

def test():
    
    # x0 = np.array([-3, -1, -3, -1])
    # diff_wood_list, symbols_wood_list = diff_wood()
    # print(g_wood(x0, diff_wood_list, symbols_wood_list))
    for m in [4, 8, 12 ,16, 20]:
        x0 = np.array([1 / m] * m)
        diff_list, symbols_list = diff_extended_powell_singular(m)
        # G, symbols_list = hess_expression(m, diff_list, symbols_list)
        # with open("cached_expression/g_extended_powell_singular_{m}.pkl".format(m=m), 'wb') as writer:
        #     pickle.dump(diff_list, writer)
        # with open("cached_expression/G_extended_powell_singular_{m}.pkl".format(m=m), 'wb') as writer:
        #     pickle.dump(G, writer)
        # with open("cached_expression/symbols_extended_powell_singular_{m}.pkl".format(m=m), 'wb') as writer:
        #     pickle.dump(symbols_list, writer)
        
        # print(g_function(x0, diff_list=diff_list, symbols_list=symbols_list))
        # print(g_EPS(x0))
        # if np.any(G_function(x0, G_lists=G, symbols_list=symbols_list)==G_EPS(x0)):
        #     print("{m} Right".format(m=m))
        if np.any(g_function(x0, diff_list=diff_list, symbols_list=symbols_list)==g_EPS(x0)):
            print("{m} Right".format(m=m))

       
    # for m in [20, 40, 60, 80 ,100]:
    #     x0 = np.array([1 / m] * m)
    #     diff_list, symbols_list = diff_trigonometric(m)
    #     G, symbols_list = hess_expression(m, diff_list, symbols_list)
    #     with open("cached_expression/g_trigonometric_{m}.pkl".format(m=m), 'wb') as writer:
    #         pickle.dump(diff_list, writer)
    #     with open("cached_expression/G_trigonometric_{m}.pkl".format(m=m), 'wb') as writer:
    #         pickle.dump(G, writer)
    #     with open("cached_expression/symbols_trigonometric_{m}.pkl".format(m=m), 'wb') as writer:
    #         pickle.dump(symbols_list, writer)
        
    #     print(g_function(x0, diff_list=diff_list, symbols_list=symbols_list))
        # print(G_function(x0, G_lists=G, symbols_list=symbols_list))
    # for m in [4]:
    #     x0 = np.array([1 / m] * m)
    #     # print(trigonometric(x0))
    #     diff_list, symbols_list = diff_trigonometric(m)
    #     G, symbols_list = hess_expression(m, diff_list, symbols_list)
    #     print(g_function(x0, diff_list=diff_list, symbols_list=symbols_list))
    #     print(g_trigonometric(x0))
        # print(G_function(x0, G_lists=G, symbols_list=symbols_list))
        # print(G_trigonometric(x0))
        

def main():
    test()

if __name__ == "__main__":
    main()