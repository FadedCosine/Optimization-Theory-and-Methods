import functions
import numpy as np
import math
import time
from goto import with_goto
import Line_Search.exact_line_search as ELS
import Line_Search.inexact_line_search as ILS
from Line_Search.GLL import GLL_search
import utils
import functools
import copy
from scipy.sparse.linalg import gmres
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
    g = gfunc(X)
    G = hess_funct(X)
    eta_pre = None
    # 把当前函数值加入func_values
    F = func(X)
    function_k += 1
    func_values.append(F)
    start_time = time.time()
    #计算下降方向d_k，这一步包括修正Hk，和计算dk = -Hk * gk
    label .count_dk
    
    #选择当前的eta
    if g_pre is None:
        eta = eta0
    else:
        if eta_mode == 1:
            # logger.info("g is {}".format(g))
            # logger.info("g size is {}".format(g.shape))
            # logger.info("g_pre size is {}".format(g_pre.shape))
            # logger.info("G_pre size is {}".format(G_pre.shape))
            # logger.info("d_pre size is {}".format(d_pre.shape))
            # logger.info("G_pre @ d_pre size is {}".format((G_pre @ d_pre).shape))
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
                eta = max(eta, gamma * eta_pre ** sigma)
    #使用GMRES方法迭代求解dk
    d = gmres(G, -g, tol=eta)[0]
    before_LS_time = time.time()
    #求得下降方向之后，此后的步骤与其他优化方法无异
    if search_mode == "ELS":
        logger.info("迭代第{iter}轮，当前函数调用次数{func_k}，当前用时{time}，当前X取值为{X}，当前g的取值为{g}, 下降方向为{d}，当前函数值为{func_x}".format(iter=k,func_k=function_k,time=before_LS_time-start_time,X=X, g=g, d=d,func_x=round(F, 8)))
        a, b, add_retreat_func = ELS.retreat_method(func, X, d, hyper_parameters=hyper_parameters["ELS"]["retreat_method"] if hyper_parameters is not None else None) 
        alpha_star, add_golden_func = ELS.golden_method(func, X, d, a, b, hyper_parameters=hyper_parameters["ELS"]["golden_method"] if hyper_parameters is not None else None) 
        add_func_k = add_retreat_func + add_golden_func
    elif search_mode == "ILS":
        logger.info("迭代第{iter}轮，当前函数调用次数{func_k}，当前用时{time}，当前X取值为{X}，当前g的取值为{g}, 下降方向为{d}，当前函数值为{func_x}".format(iter=k,func_k=function_k,time=before_LS_time-start_time,X=X, g=g, d=d,func_x=round(F, 8)))
        alpha_star, add_func_k = ILS.inexact_line_search(func, gfunc, X, d, hyper_parameters=hyper_parameters["ILS"] if hyper_parameters is not None else None) 
    elif search_mode == "GLL":
        logger.info("迭代第{iter}轮，当前函数调用次数{func_k}，当前用时{time}，当前X取值为{X}，当前g的取值为{g}, 下降方向为{d}，当前函数值为{func_x}".format(iter=k,func_k=function_k,time=before_LS_time-start_time,X=X, g=g, d=d,func_x=round(F, 8)))
        alpha_star, add_func_k, mk = GLL_search(func, gfunc, X, d, func_values, mk, hyper_parameters=hyper_parameters["GLL"] if hyper_parameters is not None else None) 
    # 更新
    logger.info("当前更新的步长为{}".format(alpha_star))
    X_new = X + d * alpha_star
    function_k = function_k + add_func_k + 1
    func_X_new = func(X_new)
    func_values.append(func_X_new)
    g_pre = g
    G_pre = G
    d_pre = d
    g = gfunc(X_new)
    G = hess_funct(X)
    
    logging.info("g is {}".format(g))
    logger.info("g的范数为{g}，epsilon * max(1, |x_k|)为{xk}".format(g = np.linalg.norm(g), xk = epsilon * max(1, np.linalg.norm(X_new))))
    # 给出的终止条件可能存在一些问题，由于编程语言进度的限制，g的下降量可能为0，从而计算 rho的时候可能存在除0的情况
    # if np.linalg.norm(g) < epsilon * max(1, np.linalg.norm(X_new)): 
    if abs(func_X_new - F) <= epsilon:
        end_time = time.time()
        logger.info("因为满足终止条件，{mode}的非精确牛顿法，迭代结束，迭代轮次{iter}，函数调用次数{func_k}，最终用时{time}，最终X={X}，最终函数值={func_X_new}".format(mode=search_mode, iter=k, func_k=function_k, time=end_time-start_time, X=X,func_X_new=func_X_new))
        return X_new, func_X_new, k, function_k
    if k > max_epoch:
        end_time = time.time()
        logger.info("超过最大迭代次数，{mode}的非精确牛顿法，迭代结束，迭代轮次{iter}，函数调用次数{func_k}，最终用时{time}，最终X={X}，最终函数值={func_X_new}".format(mode=search_mode, iter=k, func_k=function_k, time=end_time-start_time, X=X,func_X_new=func_X_new))
        return X_new, func_X_new, k, function_k
    X = X_new
    F = func_X_new
    k += 1
    goto .count_dk
            

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


if __name__ == '__main__':
    CRITERION = ["Armijo Goldstein", "Wolfe Powell", "Strong Wolfe Powell"]
    ILS_criterion = CRITERION[0]
    ELS_INM_hyper_parameters = {
        "ELS": {
            "retreat_method": {
                "a0" : 1, 
                "r": 1e-7,
                "t": 5,
            },
            "golden_method": {
                "epsilon": 1e-7,
            }
        },
        "INM": {
            "eta_mode": 1,
            "eta0": 0.5,
            "safeguard" : True,
            "gamma" : 0.9,
            "sigma" : (1 + math.sqrt(5)) / 2,
        },
        "modified_Cholesky": {
            "u": 1e-50,
        },
        "search_mode": "ELS",
        "epsilon": 1e-5,
        "max_epoch": 10000,
    }
    
    ILS_INM_hyper_parameters = {
        "ILS": {
            "rho": 0.2,
            "sigma": 0.4,
            "t": 1.5,
            "alpha0": 1e-6,
            "criterion": ILS_criterion
        },
        "GM_newton": {
            "zeta": 1e-8,
        },
        "modified_Cholesky": {
            "u": 1e-50,
        },
        "INM": {
            "eta_mode": 1,
            "eta0": 0.5,
            "safeguard" : True,
            "gamma" : 0.9,
            "sigma" : (1 + math.sqrt(5)) / 2,
        },
        "search_mode": "ILS",
        "epsilon": 1e-5,
        "max_epoch": 10000,
    }
    GLL_INM_hyper_parameters = {
        "GLL": {
            "rho": 0.25,
            "sigma": 0.4,
            "M": 5,
            "a": 1,
        },
        "modified_Cholesky": {
            "u": 1e-50,
        },
        "INM": {
            "eta_mode": 1,
            "eta0": 0.5,
            "safeguard" : True,
            "gamma" : 0.9,
            "sigma" : (1 + math.sqrt(5)) / 2,
        },
        "search_mode": "GLL",
        "epsilon": 1e-5,
        "max_epoch": 10000,
    }

    for n in [1000]:
        logger.info("Penalty1 函数")
        x0 = np.array(range(1, n + 1))
        penalty1 = functions.Penalty1(n)
        # logger.info("非精确线搜索下的FF方法")
        # X_star, func_X_star, iter_num, function_num = FF.Fletcher_Freeman(x0,  penalty1.func, penalty1.gfunc, penalty1.hess_func, hyper_parameters=GLL_LSR1_hyper_parameters)
        

        logger.info("非精确线搜索下的LSR1法") 
        X_star, func_X_star, iter_num, function_num = inexact_newton_method(x0, penalty1.func, penalty1.gfunc, penalty1.hess_func, hyper_parameters=ILS_INM_hyper_parameters)

        # logger.info("GLL线搜索下的LSR1法") 
        # X_star, func_X_star, iter_num, function_num = CLSR1(x0, penalty1.func, penalty1.gfunc, hyper_parameters=GLL_LSR1_hyper_parameters)
        

        # logger.info("Extended_Freudenstein_Roth 函数")
        # x0 = np.array([-2.] * n)
        # EFR = functions.Extended_Freudenstein_Roth(n)

        # logger.info("非精确线搜索下的FF方法")
        # X_star, func_X_star, iter_num, function_num = FF.Fletcher_Freeman(x0,  EFR.func, EFR.gfunc, EFR.hess_func, hyper_parameters=GLL_LSR1_hyper_parameters)
        
        # logger.info("精确线搜索下的LSR1法") 
        # X_star, func_X_star, iter_num, function_num = CLSR1(x0, EFR.func, EFR.gfunc, hyper_parameters=ELS_LSR1_hyper_parameters)

        # logger.info("非精确线搜索下的LSR1法") 
        # X_star, func_X_star, iter_num, function_num = CLSR1(x0, EFR.func, EFR.gfunc, hyper_parameters=ILS_LSR1_hyper_parameters)

        # logger.info("GLL线搜索下的LSR1法") 
        # X_star, func_X_star, iter_num, function_num = CLSR1(x0, EFR.func, EFR.gfunc, hyper_parameters=GLL_LSR1_hyper_parameters)
        
        # logger.info("Extended_Rosenbrock 函数")
        # ER = functions.Extended_Rosenbrock(n)
        # t = np.array(range(int(n / 2)))
        # x0[2 * t] = -1.2
        # x0[2 * t + 1] = 1
        # logger.info("精确线搜索下的LSR1法") 
        # X_star, func_X_star, iter_num, function_num = CLSR1(x0, ER.func, ER.gfunc, hyper_parameters=ELS_LSR1_hyper_parameters)

        # logger.info("非精确线搜索下的LSR1法") 
        # X_star, func_X_star, iter_num, function_num = CLSR1(x0, ER.func, ER.gfunc, hyper_parameters=ILS_LSR1_hyper_parameters)

        # logger.info("GLL线搜索下的LSR1法") 
        # X_star, func_X_star, iter_num, function_num = CLSR1(x0, ER.func, ER.gfunc, hyper_parameters=GLL_LSR1_hyper_parameters)
    
    ILS_criterion = CRITERION[1]
    ELS_LSR1_hyper_parameters = {
        "ELS": {
            "retreat_method": {
                "a0" : 1, 
                "r": 1e-7,
                "t": 5,
            },
            "golden_method": {
                "epsilon": 1e-7,
            }
        },
        "LSR1": {
            "M": 15,
        },
        "modified_Cholesky": {
            "u": 1e-20,
        },
        "search_mode": "ELS",
        "epsilon": 1e-5,
        "max_epoch": 1000,
    }
    
    ILS_LSR1_hyper_parameters = {
        "ILS": {
            "rho": 0.2,
            "sigma": 0.4,
            "t": 1.5,
            "alpha0": 1e-6,
            "criterion": ILS_criterion
        },
        "GM_newton": {
            "zeta": 1e-8,
        },
        "LSR1": {
            "M": 15,
        },
        "modified_Cholesky": {
            "u": 1e-20,
        },
        "search_mode": "ILS",
        "epsilon": 1e-5,
        "max_epoch": 1000,
    }
    GLL_LSR1_hyper_parameters = {
        "GLL": {
            "rho": 0.25,
            "sigma": 0.5,
            "M": 12,
            "a": 1,
        },
        "LSR1": {
            "M": 15,
        },
        "modified_Cholesky": {
            "u": 1e-20,
        },
        "search_mode": "GLL",
        "epsilon": 1e-5,
        "max_epoch": 1000,
    }

    x0 = np.array([1/1000] * int(1000))
    f_funciton = functions.trigonometric
    g_function = functions.g_trigonometric
    G_function = functions.G_trigonometric
    # logger.info("精确线搜索下的LSR1法") 
    # X_star, func_X_star, iter_num, function_num = CLSR1(x0, f_funciton, g_function, hyper_parameters=ELS_LSR1_hyper_parameters)

    # logger.info("非精确线搜索下的LSR1法") 
    # X_star, func_X_star, iter_num, function_num = CLSR1(x0, f_funciton, g_function, hyper_parameters=ILS_LSR1_hyper_parameters)

    # logger.info("GLL线搜索下的LSR1法") 
    # X_star, func_X_star, iter_num, function_num = CLSR1(x0, f_funciton, g_function, hyper_parameters=GLL_LSR1_hyper_parameters)