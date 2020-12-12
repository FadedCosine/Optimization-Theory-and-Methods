import functions
import numpy as np
import math
from goto import with_goto
import Line_Search.exact_line_search as ELS
import Line_Search.inexact_line_search as ILS
from Line_Search.GLL import GLL_search
import utils
import functools
import copy
import scipy
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

@with_goto
def inexact_newton_method(X, func, gfunc, hess_funct, hyper_parameters=None, search_mode="ELS", eta_mode=1, safeguard=True, eta0=0.5, gamma=1, sigma=1.5, epsilon=1e-5, max_epoch=1000):
    """[使用非精确牛顿法极小值点
         d = -G_k^{-1} * g_k]

    Args:
        X ([np.array]): [Input X]
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        hess_funct ([回调函数]): [目标函数的Hessian矩阵]
        hyper_parameters: (Dic): 超参数，超参数中包括：
            search_mode (str, optional): [线搜索的模式（选择精确线搜索还是非精确线搜索）]. Defaults to 'ELS'. ['ELS', 'ILS']
            eta_mode (int, optional): [{eta}选择的方式]. Defaults to 1. [1, 2]
            eta0 ([float], optional): [eta的初值]. Defaults to 0.5.
            gamma ([float], optional): [eta选择2当中的系数参数]. Defaults to 1.
            sigma ([float], optional): [eta选择2当中的指数参数]. Defaults to 1.5.
            safeguard ([bool], optional): [是否使用安全保护]. Defaults to True.
            epsilon ([float], optional): [当函数值下降小于epsilon，迭代结束]. Defaults to 1e-5.
            max_epoch (int, optional): [最大允许的迭代次数]. Defaults to 1000.

    Returns:
        返回求解得到的极小值点，极小值点对应的函数值和迭代次数
    """
    if hyper_parameters is not None:
        search_mode = hyper_parameters["search_mode"]
        epsilon = hyper_parameters["epsilon"]
        max_epoch = hyper_parameters["max_epoch"]
        eta_mode = hyper_parameters["INM"]["eta_mode"]
        eta0 = hyper_parameters["INM"]["eta0"]
        safeguard = hyper_parameters["INM"]["safeguard"]
        if eta_mode == 2:
            gamma = hyper_parameters["INM"]["gamma"]
            sigma = hyper_parameters["INM"]["sigma"]

    n = len(X)
    k = 1
    function_k = 0
    func_values = [] # 记录每一步的函数值，在GLL中有用
    mk = 0 # GLL当中的mk初始值
    g_pre = None
    G_pre = None
    d_pre = None
    eta_pre = None
    # 把当前函数值加入func_values
    F = func(X)
    function_k += 1
    func_values.append(F)
    start_time = time.time()
    #计算下降方向d_k，这一步包括修正Hk，和计算dk = -Hk * gk
    label .count_dk
    g = gfunc(X)
    G = hess_funct(X)
    #选择当前的eta
    if g_pre is None:
        eta = eta0
    else:
        if eta_mode == 1:
            eta = np.linalg.norm(g - g_pre - G_pre @ d_pre) / np.linalg.norm(g_pre)
        elif eta_mode == 2:
            eta = gamma * (np.linalg.norm(g) / np.linalg.norm(g_pre)) ** sigma
        
    # 安全保护
    if eta_pre is not None and safeguard:
        if eta_mode == 1:
            if eta_pre ** ((1/math.sqrt(5))/2) > 0.1:
                eta = max(eta, eta_pre ** ((1/math.sqrt(5))/2) )
        elif eta_mode == 2:
            if gamma * eta_pre ** sigma > 0.1:
                eta = max(eta, gamma * eta_pre ** sigma )
    
            







# @with_goto
# def INBM(X, func, gfunc, hess_funct, hyper_parameters=None, search_mode="ELS", t=1e-4, eta_max=0.9, theta_min=0.1, theta_max=0.5, epsilon=1e-5, max_epoch=1000):
"""[summary]

Args:
    X ([np.array]): [Input X]
    func ([回调函数]): [目标函数]
    gfunc ([回调函数]): [目标函数的一阶导函数]
    hess_funct ([回调函数]): [目标函数的Hessian矩阵]
    hyper_parameters: (Dic): 超参数，超参数中包括：
        search_mode (str, optional): [description]. Defaults to "ELS".
        t ([type], optional): [description]. Defaults to 1e-4.
        eta_max (float, optional): [description]. Defaults to 0.9.
        theta_min (float, optional): [description]. Defaults to 0.1.
        theta_max (float, optional): [description]. Defaults to 0.5.
        epsilon ([type], optional): [description]. Defaults to 1e-5.
        max_epoch (int, optional): [description]. Defaults to 1000.
"""