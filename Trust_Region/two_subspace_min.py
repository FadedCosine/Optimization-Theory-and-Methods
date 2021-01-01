import functions
import numpy as np
import math
import time
from goto import with_goto
import utils
import functools
import copy
import random
from utils import is_pos_def
from scipy import optimize
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def two_subspace_min(X, func, gfunc, hess_func, delta, hyper_parameters=None, v_0=1e-2, epsilon=1e-10, max_epoch=10000):
    """ 二维子空间极小化方法 求解TR子问题，注意此方法不是迭代方法

    Args:
        X ([np.array]): [Input X]
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        hess_func ([回调函数]): [目标函数的Hessian矩阵]
        delta ([float]): [TR子问题约束中的delta]
        hyper_parameters: (Dic): 超参数，超参数中包括：
            v_0 ([float]], optional): [v的初值]. Defaults to 1e-2.
            epsilon ([float], optional): [解决浮点数相减不精确的问题，用于判等]. Defaults to 1e-10.
            max_epoch (int, optional): [description]. Defaults to 1000.

    Returns:
        [type]: [description]
    """
    k = 0
    function_k = 0

    I = np.identity(len(X))
    G = hess_func(X)
    
    # 先判断G是否正定
    if not is_pos_def(G): # G非正定的情况，通过取 v \in {-lambda_n , -2 * lambda_n}，使得G正定
        values, _vector = np.linalg.eig(G)
        values = sorted(values)
        lambda_n = values[0]
        # v = random.uniform(-lambda_n, -2 * lambda_n)
        v = - 3 / 2 * lambda_n
        G = G + v * I
    
    inv_G = np.linalg.inv(G) 
    g = gfunc(X)
    abs_g = np.linalg.norm(g)
    inv_G_g = inv_G @ g

    g_tilde = np.array([abs_g**2, g @ inv_G @ g], dtype=float)
    G_tilde = np.array(
        [[g @ G @ g, abs_g**2     ],
            [abs_g**2,  g @ inv_G @ g]],
            dtype=float)
    G_overline = 2 * np.array(
        [[abs_g**2, g @ inv_G @ g],
            [g @ inv_G @ g, np.linalg.norm(inv_G_g) ** 2]],
            dtype=float)
    inv_G_tilde = np.linalg.inv(G_tilde) 
    u_star = - inv_G_tilde @ g_tilde
    if 1/2 * (u_star @ G_overline @ u_star) <= delta ** 2:
        u = u_star

    else:
        def fun(x):
            input = - np.linalg.inv(G_tilde + x * G_overline) @ g_tilde
            return [1/2 * (input @ G_overline @ input) - delta ** 2]
        lambda_sol = optimize.root(fun, [0])
        lambda_ = float(lambda_sol.x)
        u = - np.linalg.inv(G_tilde + lambda_ * G_overline) @ g_tilde
    alpha = u[0]
    beta = u[1]
    d = alpha * g + beta * inv_G_g
    return d, function_k


  