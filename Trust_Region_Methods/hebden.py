import numpy as np
from goto import with_goto
from utils import is_pos_def
import copy
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

@with_goto
def hebden(X, func, gfunc, hess_func, delta, hyper_parameters=None, v_0=1e-2, epsilon=1e-10, max_epoch=1000):
    """ Hebden方法求解TR子问题

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
    k = 1
    v_k = v_0

    I = np.identity(len(X))
    G_org = hess_func(X)
    G = copy.deepcopy(G_org)
    # 先判断G是否正定

    if not is_pos_def(G): # G非正定的情况，通过取 v \in {-lambda_n , -2 * lambda_n}，使得G正定
        values, _vector = np.linalg.eig(G)
        values = sorted(values)
        lambda_n = values[0]
        # v = random.uniform(-lambda_n, -2 * lambda_n)
        v_k = - 3 / 2 * lambda_n
        G = G + v_k * I
 
    inv_G = np.linalg.inv(G) 
    g = gfunc(X)
    d_v = - inv_G @ g
    
    if np.linalg.norm(d_v) < delta:
        return d_v, k

    label.step4
    abs_d_v = np.linalg.norm(d_v)
    d_v_prime = - inv_G @ d_v # d_v的导数
    phi_v = abs_d_v - delta 
    phi_v_prime = d_v @ d_v_prime / abs_d_v # phi_v的导数
    # 判断终止准则是否成立
    if abs(phi_v) <= epsilon:
        return d_v, k
    if k > max_epoch:
        raise Exception("使用Hebden方法求解TR子问题时，超过最大迭代次数：%d", max_epoch)
    # return d_v, k

    # 更新v_k
    v_k = v_k - (phi_v + delta) * phi_v / (delta * phi_v_prime)
    # 重新计算(G+vI)
    G = G_org + v_k * I
    inv_G = np.linalg.inv(G)  # 求Hesse矩阵的逆
    d_v = - inv_G @ g
    k = k + 1
    goto.step4
