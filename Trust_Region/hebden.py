import copy
import numpy as np
from goto import with_goto
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

@with_goto
def Hebden_method(X, func, gfunc, hess_func, delta, hyper_parameters=None, v_0=1e-2, epsilon=1e-10, max_epoch=10000):
    """ Hebden方法求解TR子问题

    Args:
        X ([np.array]): [Input X]
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        hess_func ([回调函数]): [目标函数的Hessian矩阵]
        delta ([float]): [TR子问题约束中的delta]
        hyper_parameters: (Dic): 超参数，超参数中包括：
            v_0 ([float]], optional): [v的初值]. Defaults to 1e-2.
            epsilon ([float], optional): [解决浮点数相减不精确的问题，用于判等]. Defaults to 1e-6.
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
    d_v = -np.matmul(inv_G, g)
    
    if np.linalg.norm(d_v) < delta:
        return d_v, function_k

    label.step4
    abs_d_v = np.linalg.norm(d_v)
    d_v_prime = -np.matmul(inv_G, d_v) # d_v的导数
    phi_v = abs_d_v - delta 
    phi_v_prime = d_v @ d_v_prime / abs_d_v # phi_v的导数
    # 判断终止准则是否成立
    if abs(phi_v) <= epsilon:
        return d_v, function_k
    if k > max_epoch:
        logger.info("求解TR子问题时，超过最大迭代次数：%d", max_epoch)
        return d_v, function_k

    # 更新v_k
    v_k = v_k - (phi_v + delta) * phi_v / (delta * phi_v_prime)
    # 重新计算(G+vI)
    G = G_org + v_k * I
    inv_G = np.linalg.inv(G)  # 求Hesse矩阵的逆
    d_v = -np.matmul(inv_G, g)
    k = k + 1
    goto.step4

@with_goto
def Hebden_method_2(x0, function, diff_function, hesse_function, delta_k, v_0=1e-2, epsilon=1e-10, max_epoch=10000, parameters=None):
    # 初始化
    k = 0
    fx_k = 0
    x = copy.deepcopy(x0)
    v_k = copy.deepcopy(v_0)

    label.step2
    G = hesse_function(x)
    length = G.shape[0]
    G = G + v_k * np.identity(length)
    values, _vector = np.linalg.eig(G)
    for value in values:
        if value <= 0:
            v_k = v_k * 4
            goto.step2

    G_1 = np.linalg.inv(G)  # 求Hesse矩阵的逆
    g = diff_function(x)
    newton_descent = -np.matmul(G_1, g)
    d_2 = np.linalg.norm(newton_descent)
    if d_2 < delta_k:
        return newton_descent, fx_k

    label.step4
    d_v = copy.deepcopy(newton_descent)
    d_v_diff = -np.matmul(G_1, d_v)
    phi_v = np.linalg.norm(d_v) - delta_k
    phi_v_diff = np.matmul(d_v.T, d_v_diff) / np.linalg.norm(d_v)
    # 判断终止准则是否成立
    if abs(phi_v) <= epsilon * delta_k:
        return d_v, fx_k
    if k > max_epoch:
        raise Exception("超过最大迭代次数！！！")
    # 更新v_k
    v_k = v_k - (phi_v + delta_k) * phi_v / (delta_k * phi_v_diff)
    # 重新计算(G+vI)
    G = hesse_function(x)
    length = G.shape[0]
    G = G + v_k * np.identity(length)
    G_1 = np.linalg.inv(G)  # 求Hesse矩阵的逆
    g = diff_function(x)
    newton_descent = -np.matmul(G_1, g)
    k = k + 1
    goto.step4
