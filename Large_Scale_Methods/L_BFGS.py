import numpy as np
import Line_Search.exact_line_search as ELS
import Line_Search.inexact_line_search as ILS
from Line_Search.GLL import GLL_search
import Newton_Methods.fletcher_freeman as FF
import Newton_Methods.newton_method as nm
from goto import with_goto
import logging
import functions
import copy
import time
from queue import Queue

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

@with_goto
def LBFGS(X, func, gfunc, hyper_parameters=None, M = 15, search_mode="ELS", epsilon=1e-5, max_epoch=1000):
    """ 有限内存的BFGS方法

    Args:
        X ([np.array]): [Input X]
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        hess_func ([回调函数]): [目标函数的Hessian矩阵]
        hyper_parameters: (json): 超参数，超参数中包括：
            M (int, optional): [计算修正Hk的时候，需要之前记录的M个信息，记录的信息包括s和y], 要求M的取值范围在[5, 9, 15]. Defaults to 15.
            search_mode (str, optional): [线搜索的模式（选择精确线搜索还是非精确线搜索）]. Defaults to 'ELS'. ['ELS', 'ILS']
            epsilon ([float], optional): [||g_k|| < 1e-5 * max(1, ||x_k||)时，迭代结束]. Defaults to 1e-8.
            max_epoch (int, optional): [最大允许的迭代次数]. Defaults to 1000.
    """
    if hyper_parameters is not None:
        M = hyper_parameters["LBFGS"]["M"]
        search_mode = hyper_parameters["search_mode"]
        epsilon = hyper_parameters["epsilon"]
        max_epoch = hyper_parameters["max_epoch"]
    n = len(X)
    k = 0
    function_k = 0
    func_values = [] # 记录每一步的函数值，在GLL中有用
    mk = 0 # GLL当中的mk初始值

    s_history = [] # 记录每一步的x_{k+1} - x_{k}，LBFGS修正Hk时有用
    y_history = [] # 记录每一步的g_{k+1} - g_{k}，LBFGS修正Hk时有用
    p_history = [] # 1 / s_{k}^ t * y_k，以免重复计算 
    LBFGS_alpha = np.zeros(M) # LBFGS算法1中计算的ai，先声明，反复使用节约内存空间
    g = gfunc(X)
    # 把当前函数值加入func_values
    F = func(X)
    function_k += 1
    func_values.append(F)
    start_time = time.time()
    #计算下降方向d_k，这一步包括修正Hk，和计算dk = -Hk * gk
    label .count_dk

    #使用LBFGS得到Hk
    # LBFGS 的 算法1，计算ai
    q = copy.deepcopy(g)
    for i in range(min(len(s_history), M)):
        LBFGS_alpha[M - 1 - i] = p_history[-i - 1] * (s_history[-i - 1] @ q)
        q -= LBFGS_alpha[M - 1 - i] * y_history[-i - 1]
    # LBFGS 的 算法2，计算r = Hk gk
    # if len(p_history) > 0:
    #     Hk_0 = np.eye(n, dtype=float) * ((s_history[-1] @ y_history[-1])/ (y_history[-1] @ y_history[-1]))
    # else:
    Hk_0 = np.eye(n, dtype=float)
    r = Hk_0 @ q 
    for i in range(min(len(s_history), M), 0, -1):
        beta = p_history[-i] * (y_history[-i] @ r)
        r += (LBFGS_alpha[-i] - beta) * s_history[-i]
    
    d = np.array(-r)
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
        
    logger.info("当前更新的步长为{}".format(alpha_star))
    X_new = X + d * alpha_star
    function_k = function_k + add_func_k + 1
    func_X_new = func(X_new)
    func_values.append(func_X_new)
    g_new = gfunc(X_new)

    
    s_history.append(d * alpha_star)
    y_history.append(g_new - g)
    p_history.append(1 / (s_history[-1] @ y_history[-1]))

    # 更新
    logging.info("g is {}".format(g_new))
    logger.info("g的范数为{g}，epsilon * max(1, |x_k|)为{xk}".format(g = np.linalg.norm(g_new), xk = epsilon * max(1, np.linalg.norm(X_new))))
    # 给出的终止条件可能存在一些问题，由于编程语言进度的限制，g的下降量可能为0，从而计算 rho的时候可能存在除0的情况
    if np.linalg.norm(g_new) < epsilon * max(1, np.linalg.norm(X_new)): 
    # if abs(func_X_new - F) <= epsilon:
        end_time = time.time()
        logger.info("因为满足终止条件，{mode}的有限内存BFGS方法，迭代结束，迭代轮次{iter}，函数调用次数{func_k}，最终用时{time}，最终X={X}，最终函数值={func_X_new}".format(mode=search_mode, iter=k, func_k=function_k, time=end_time-start_time, X=X,func_X_new=func_X_new))
        return X_new, func_X_new, k, function_k
    if k > max_epoch:
        end_time = time.time()
        logger.info("超过最大迭代次数，{mode}的有限内存BFGS方法，迭代结束，迭代轮次{iter}，函数调用次数{func_k}，最终用时{time}，最终X={X}，最终函数值={func_X_new}".format(mode=search_mode, iter=k, func_k=function_k, time=end_time-start_time, X=X,func_X_new=func_X_new))
        
        return X_new, func_X_new, k, function_k
    X = X_new
    g = g_new
    F = func_X_new
    k += 1
    goto .count_dk


@with_goto
def CLBFGS(X, func, gfunc, hyper_parameters=None, M = 15, search_mode="ELS", epsilon=1e-5, max_epoch=1000):
    """ 压缩形式的有限内存BFGS方法

    Args:
        X ([np.array]): [Input X]
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        hess_func ([回调函数]): [目标函数的Hessian矩阵]
        hyper_parameters: (json): 超参数，超参数中包括：
            M (int, optional): [计算修正Hk的时候，需要之前记录的M个信息，记录的信息包括s和y], 要求M的取值范围在[5, 9, 15]. Defaults to 15.
            search_mode (str, optional): [线搜索的模式（选择精确线搜索还是非精确线搜索）]. Defaults to 'ELS'. ['ELS', 'ILS']
            epsilon ([float], optional): [||g_k|| < 1e-5 * max(1, ||x_k||)时，迭代结束]. Defaults to 1e-8.
            max_epoch (int, optional): [最大允许的迭代次数]. Defaults to 1000.
    """
    if hyper_parameters is not None:
        M = hyper_parameters["LBFGS"]["M"]
        search_mode = hyper_parameters["search_mode"]
        epsilon = hyper_parameters["epsilon"]
        max_epoch = hyper_parameters["max_epoch"]
    n = len(X)
    k = 0
    function_k = 0
    func_values = [] # 记录每一步的函数值，在GLL中有用
    mk = 0 # GLL当中的mk初始值

    Sk_que = Queue() # 记录最多M个s_k，LBFGS修正Hk时有用
    Yk_que = Queue() # 记录最多M个y_k，LBFGS修正Hk时有用
    Dk_que = Queue() # 记录最多M个s^T * y

    g = gfunc(X)
    F = func(X)
    function_k += 1
    func_values.append(F)
    start_time = time.time()
    #计算下降方向d_k，这一步包括使用压缩形式修正Hk，和计算dk = -Hk * gk
    label .count_dk

    # if len(p_history) > 0:
    #     mu = ((s_history[-1] @ y_history[-1])/ (y_history[-1] @ y_history[-1]))
    # else:
    #     mu = 1
    Hk = np.eye(n, dtype=float) 
    item_num = min(Sk_que.qsize(), M)
    if item_num > 0:
        Sk = np.mat(Sk_que.queue).T
        Yk = np.mat(Yk_que.queue).T 
        Rk = np.zeros((item_num, item_num), dtype=float)
        for i in range(item_num):
            for j in range(i, item_num):
                Rk[i][j] = Sk_que.queue[i] @ Yk_que.queue[j]
        
        Rk_inv = np.linalg.inv(Rk)
        Dk = np.diag(Dk_que.queue)
        mid_mat11 = Rk_inv.T @ (Dk + Yk.T @ Hk @ Yk) @ Rk_inv
        mid_mat_left = np.vstack((mid_mat11, -Rk_inv))
        mid_mat_right = np.vstack((-Rk_inv.T, np.zeros((item_num, item_num), dtype=float)))
        mid_mat = np.hstack((mid_mat_left, mid_mat_right))
        Hk = Hk + np.hstack((Sk, Hk @ Yk)) @ mid_mat @ np.vstack((Sk.T, Yk.T @ Hk))

    d = np.squeeze(np.array(-Hk @ g))
   
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
        
    logger.info("当前更新的步长为{}".format(alpha_star))
    X_new = X + d * alpha_star
    function_k = function_k + add_func_k + 1
    func_X_new = func(X_new)
    func_values.append(func_X_new)
    g_new = gfunc(X_new)

    if item_num == M:
        Sk_que.get()
        Yk_que.get()
        Dk_que.get()
    Sk_que.put(d * alpha_star)
    Yk_que.put(g_new - g)
    Dk_que.put((d * alpha_star) @ (g_new - g))

    # 更新
    logging.info("g is {}".format(g_new))
    logger.info("g的范数为{g}，epsilon * max(1, |x_k|)为{xk}".format(g = np.linalg.norm(g_new), xk = epsilon * max(1, np.linalg.norm(X_new))))
    # 给出的终止条件可能存在一些问题，由于编程语言进度的限制，g的下降量可能为0，从而计算 rho的时候可能存在除0的情况
    if np.linalg.norm(g_new) < epsilon * max(1, np.linalg.norm(X_new)): 
    # if abs(func_X_new - F) <= epsilon:
        end_time = time.time()
        logger.info("因为满足终止条件，{mode}的有限内存BFGS方法，迭代结束，迭代轮次{iter}，函数调用次数{func_k}，最终用时{time}，最终X={X}，最终函数值={func_X_new}".format(mode=search_mode, iter=k, func_k=function_k, time=end_time-start_time, X=X,func_X_new=func_X_new))
        return X_new, func_X_new, k, function_k
    if k > max_epoch:
        end_time = time.time()
        logger.info("超过最大迭代次数，{mode}的有限内存BFGS方法，迭代结束，迭代轮次{iter}，函数调用次数{func_k}，最终用时{time}，最终X={X}，最终函数值={func_X_new}".format(mode=search_mode, iter=k, func_k=function_k, time=end_time-start_time, X=X,func_X_new=func_X_new))
        
        return X_new, func_X_new, k, function_k
    X = X_new
    g = g_new
    
    F = func_X_new
    k += 1
    goto .count_dk


if __name__ == '__main__':
    CRITERION = ["Armijo Goldstein", "Wolfe Powell", "Strong Wolfe Powell"]
    ILS_criterion = CRITERION[0]
    ELS_LBFGS_hyper_parameters = {
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
        "LBFGS": {
            "M": 15,
        },
        "search_mode": "ELS",
        "epsilon": 1e-5,
        "max_epoch": 1000,
    }
    
    ILS_LBFGS_hyper_parameters = {
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
        "LBFGS": {
            "M": 15,
        },
        "search_mode": "ILS",
        "epsilon": 1e-5,
        "max_epoch": 1000,
    }
    GLL_LBFGS_hyper_parameters = {
        "GLL": {
            "rho": 0.25,
            "sigma": 0.4,
            "M": 15,
            "a": 1,
        },
        "LBFGS": {
            "M": 10,
        },
        "search_mode": "GLL",
        "epsilon": 1e-5,
        "max_epoch": 1000,
    }

    for n in [100]:
        # logger.info("Penalty1 函数")
        # x0 = np.array(range(1, n + 1))
        # penalty1 = functions.Penalty1(n)
        # logger.info("非精确线搜索下的LBFGS法") 
        # X_star, func_X_star, iter_num, function_num = LBFGS(x0, penalty1.func, penalty1.gfunc, hyper_parameters=ILS_LBFGS_hyper_parameters)

        # logger.info("GLL线搜索下的LBFGS法") 
        # X_star, func_X_star, iter_num, function_num = LBFGS(x0, penalty1.func, penalty1.gfunc, hyper_parameters=GLL_LBFGS_hyper_parameters)
        

        logger.info("Extended_Freudenstein_Roth 函数")
        x0 = np.array([-2.] * n)
        EFR = functions.Extended_Freudenstein_Roth(n)

        # logger.info("非精确线搜索下的FF方法")
        # X_star, func_X_star, iter_num, function_num = FF.Fletcher_Freeman(x0,  EFR.func, EFR.gfunc, EFR.hess_func, hyper_parameters=GLL_LBFGS_hyper_parameters)
        
        # logger.info("精确线搜索下的LBFGS法") 
        # X_star, func_X_star, iter_num, function_num = LBFGS(x0, EFR.func, EFR.gfunc, hyper_parameters=ELS_LBFGS_hyper_parameters)

        logger.info("非精确线搜索下的LBFGS法") 
        X_star, func_X_star, iter_num, function_num = LBFGS(x0, EFR.func, EFR.gfunc, hyper_parameters=ILS_LBFGS_hyper_parameters)

        # logger.info("GLL线搜索下的LBFGS法") 
        # X_star, func_X_star, iter_num, function_num = CLBFGS(x0, EFR.func, EFR.gfunc, hyper_parameters=GLL_LBFGS_hyper_parameters)
        
        # logger.info("Extended_Rosenbrock 函数")
        # ER = functions.Extended_Rosenbrock(n)
        # t = np.array(range(int(n / 2)))
        # x0[2 * t] = -1.2
        # x0[2 * t + 1] = 1
        # logger.info("精确线搜索下的LBFGS法") 
        # X_star, func_X_star, iter_num, function_num = CLBFGS(x0, ER.func, ER.gfunc, hyper_parameters=ELS_LBFGS_hyper_parameters)

        # logger.info("非精确线搜索下的LBFGS法") 
        # X_star, func_X_star, iter_num, function_num = CLBFGS(x0, ER.func, ER.gfunc, hyper_parameters=ILS_LBFGS_hyper_parameters)

        # logger.info("GLL线搜索下的LBFGS法") 
        # X_star, func_X_star, iter_num, function_num = CLBFGS(x0, ER.func, ER.gfunc, hyper_parameters=GLL_LBFGS_hyper_parameters)
    
    ILS_criterion = CRITERION[1]
    ELS_LBFGS_hyper_parameters = {
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
        "LBFGS": {
            "M": 15,
        },
        "search_mode": "ELS",
        "epsilon": 1e-5,
        "max_epoch": 1000,
    }
    
    ILS_LBFGS_hyper_parameters = {
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
        "LBFGS": {
            "M": 15,
        },
        "search_mode": "ILS",
        "epsilon": 1e-5,
        "max_epoch": 1000,
    }
    GLL_LBFGS_hyper_parameters = {
        "GLL": {
            "rho": 0.25,
            "sigma": 0.5,
            "M": 12,
            "a": 1,
        },
        "LBFGS": {
            "M": 15,
        },
        "search_mode": "GLL",
        "epsilon": 1e-5,
        "max_epoch": 1000,
    }

    # x0 = np.array([1/1000] * int(1000))
    # f_funciton = functions.trigonometric
    # g_function = functions.g_trigonometric
    # G_function = functions.G_trigonometric
    # logger.info("精确线搜索下的LBFGS法") 
    # X_star, func_X_star, iter_num, function_num = LBFGS(x0, f_funciton, g_function, hyper_parameters=ELS_LBFGS_hyper_parameters)

    # logger.info("非精确线搜索下的LBFGS法") 
    # X_star, func_X_star, iter_num, function_num = LBFGS(x0, f_funciton, g_function, hyper_parameters=ILS_LBFGS_hyper_parameters)

    # logger.info("GLL线搜索下的LBFGS法") 
    # X_star, func_X_star, iter_num, function_num = LBFGS(x0, f_funciton, g_function, hyper_parameters=GLL_LBFGS_hyper_parameters)