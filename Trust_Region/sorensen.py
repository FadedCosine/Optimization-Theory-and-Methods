import functions
import numpy as np
import math
import time
from goto import with_goto
import utils
import functools
import copy
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


@with_goto
def sorensen(X, func, gfunc, hess_func, delta, hyper_parameters=None, v_0=1e-2, epsilon=1e-10, max_epoch=10000):
    """ sorensen求解TR子问题

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
    # 初始化
    k = 0
    function_k = 0
    v_k = v_0

    I = np.identity(len(X))
    G_org = hess_func(X)
    G = G_org
    eigenvalue, eigen_vector = np.linalg.eig(G)
    # G非正定
    while np.any(eigenvalue <= 0):
        G = G + v_k * I
        v_k = v_k * 4
        eigenvalue, eigen_vector = np.linalg.eig(G)
    # 此时的G为 G0 + vk I
    
    inv_G = np.linalg.inv(G) 
    g = gfunc(X)
    
    d_v = - inv_G @ g
    if np.linalg.norm(d_v) < delta:
        return d_v, function_k

    label.step4
    L = np.linalg.cholesky(G)
    inv_L = np.linalg.inv(L) 
    p_l = - inv_G @ g
    q_l = inv_L @ p_l
    abs_d_v = np.linalg.norm(d_v)
    phi_v = abs_d_v - delta 

    # 判断终止准则是否成立
    if abs(phi_v) <= epsilon:
        return d_v, function_k
    if k > max_epoch:
        raise Exception("使用sorensen方法求解TR子问题时，超过最大迭代次数：%d", max_epoch)

    # 更新v_k
    abs_p_l = np.linalg.norm(p_l)
    abs_q_l = np.linalg.norm(q_l)
    v_k = v_k + ((abs_p_l/abs_q_l) ** 2) * (abs_p_l - delta) / delta
    # 重新计算(G+vI)
    G = G_org + v_k * I
    inv_G = np.linalg.inv(G)  # 求Hesse矩阵的逆
    d_v = -np.matmul(inv_G, g)
    k = k + 1
    goto.step4
