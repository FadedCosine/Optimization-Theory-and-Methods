import functions
import exact_line_search as ELS
import inexact_line_search as ILS
import fletcher_freeman as FF
import newton_method 
import utils
import numpy as np
import functools
import pickle
import argparse
from multiprocessing.pool import Pool
import os
import multiprocessing
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
"""
精确线搜索超参数:
    进退法：
        r:  步长更新的步长
        t:  >1的步长放大率
    0.618法：
        epsilon：   终止条件阈值

非精确线搜索超参数：
    rho： Armijo准则中的参数, range in (0, 1/2).
    sigma： Wolfe准则中的参数, range in (rho, 1).

阻尼牛顿法超参数:
    use_modified_Cholesky:  是否使用修正Cholesky分解计算下降方向

GM稳定牛顿法超参数：
    zeta：  当gk的模大于zeta， 求解方程得到下降方向

修正Cholesky分解超参数：
    u： 机器精度

对于任何最优化算法来说都的超参数：
    search_mode：   线搜索方法，从["ELS", "ILS"]中选择
    epsilon：   当函数值下降小于epsilon，迭代结束
    max_epoch： 最大允许的迭代次数

"""
hyper_parameters = {
    "ELS": {
        "retreat_method": {
            "r": 1e-5,
            "t": 1.5,
        },
        "golden_method": {
            "epsilon": 1e-5,
        }
    },
    "ILS": {
        "rho": 0.1,
        "sigma": 0.4,
        "t": 5,
        "criterion": "Wolfe Powell"
    },
    "damp_newton": {
        "use_modified_Cholesky" : True,
    },
    "GM_newton": {
        "zeta": 1e-2,
    },
    "modified_Cholesky": {
        "u": 1e-10,
    },
    "search_mode": "ELS",
    "epsilon": 1e-8,
    "max_epoch": 1000,
}

x0 = np.array([-3, -1, -3, -1])
diff_wood_list, symbols_wood_list = functions.diff_wood_expression()
g_wood_partial = functools.partial(functions.g_wood, diff_list=diff_wood_list, symbols_list=symbols_wood_list)
hess_wood_lists, symbols_wood_list = functions.hess_wood_expression()
G_wood_partial = functools.partial(functions.G_wood, G_lists=hess_wood_lists, symbols_list=symbols_wood_list)


# logger.info("精确线搜索下的阻尼牛顿法") # 使用基本牛顿法的下降方向无法收敛，使用修正Cholesky分解的下降方向可以收敛
ELS_damp_newton_hyper_parameters = {
    "ELS": {
        "retreat_method": {
            "a0" : 1e-2 ,
            "r": 1e-6,
            "t": 5,
        },
        "golden_method": {
            "epsilon": 1e-8,
        }
    },
    "damp_newton": {
        "use_modified_Cholesky" : True,
    },
    "modified_Cholesky": {
        "u": 1e-10,
    },
    "search_mode": "ELS",
    "epsilon": 1e-8,
    "max_epoch": 1000,
}
# X_star, func_X_star, iter_num = newton_method.damp_newton(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ELS_damp_newton_hyper_parameters)


# logger.info("非精确线搜索下的阻尼牛顿法") # 可收敛
ILS_damp_newton_hyper_parameters = {
    "ILS": {
        "rho": 0.1,
        "sigma": 0.4,
        "t": 5,
        "alpha0": 1e-6,
        "criterion": "Armijo Goldstein"
    },
    "damp_newton": {
        "use_modified_Cholesky" : False,
    },
    "modified_Cholesky": {
        "u": 1e-10,
    },
    "search_mode": "ILS",
    "epsilon": 1e-8,
    "max_epoch": 1000,
}
# X_star, func_X_star, iter_num = newton_method.damp_newton(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ILS_damp_newton_hyper_parameters)

# logger.info("GLL线搜索下的阻尼牛顿法") # 可收敛
GLL_damp_newton_hyper_parameters = {
    "GLL": {
        "rho": 0.5,
        "sigma": 0.5,
        "M": 10,
        "a": 10**5,
    },
    "damp_newton": {
        "use_modified_Cholesky" : False,
    },
    "search_mode": "GLL",
    "epsilon": 1e-8,
    "max_epoch": 1000,
}
# X_star, func_X_star, iter_num = newton_method.damp_newton(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=GLL_damp_newton_hyper_parameters)

# logger.info("精确线搜索下的GM稳定牛顿法") # 可收敛
ELS_GM_newton_hyper_parameters = {
    "ELS": {
        "retreat_method": {
            "r": 1e-5,
            "t": 1.5,
        },
        "golden_method": {
            "epsilon": 1e-8,
        }
    },
    "GM_newton": {
        "zeta": 1e-2,
    },
    "modified_Cholesky": {
        "u": 1e-10,
    },
    "search_mode": "ELS",
    "epsilon": 1e-8,
    "max_epoch": 1000,
}
# X_star, func_X_star, iter_num = newton_method.GM_newton(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ELS_GM_newton_hyper_parameters)

# logger.info("非精确线搜索下的GM稳定牛顿法") # 可收敛
ILS_GM_newton_hyper_parameters = {
    "ILS": {
        "rho": 0.1,
        "sigma": 0.4,
        "t": 5,
        "alpha0": 1e-6,
        "criterion": "Wolfe Powell"
    },
    "GM_newton": {
        "zeta": 1e-2,
    },
    "modified_Cholesky": {
        "u": 1e-10,
    },
    "search_mode": "ILS",
    "epsilon": 1e-8,
    "max_epoch": 1000,
}
# X_star, func_X_star, iter_num = newton_method.GM_newton(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ILS_GM_newton_hyper_parameters)

# logger.info("GLL线搜索下的GM牛顿法") # 可收敛
GLL_GM_newton_hyper_parameters = {
    "GLL": {
        "rho": 0.5,
        "sigma": 0.5,
        "M": 10,
        "a": 10**5,
    },
        "GM_newton": {
        "zeta": 1e-2,
    },
        "modified_Cholesky": {
        "u": 1e-10,
    },
    "search_mode": "GLL",
    "epsilon": 1e-8,
    "max_epoch": 1000,
}

# X_star, func_X_star, iter_num = newton_method.GM_newton(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=GLL_GM_newton_hyper_parameters)

# logger.info("精确线搜索下的FF方法") # 可收敛
ELS_FF_hyper_parameters = {
    "ELS": {
        "retreat_method": {
            "a0": 1e-2, 
            "r": 1e-4,
            "t": 10,
        },
        "golden_method": {
            "epsilon": 1e-8,
        }
    },
    "search_mode": "ELS",
    "epsilon": 1e-8,
    "max_epoch": 10000,
}
# X_star, func_X_star, iter_num = FF.Fletcher_Freeman(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ELS_FF_hyper_parameters)
# logger.info("非精确线搜索下的FF方法") # 可收敛
ILS_FF_hyper_parameters = {
    "ILS": {
        "rho": 0.1,
        "sigma": 0.4,
        "t": 5,
        "alpha0": 1e-6,
        "criterion": "Armijo Goldstein"
    },
    "search_mode": "ILS",
    "epsilon": 1e-8,
    "max_epoch": 1000,
}

# X_star, func_X_star, iter_num = FF.Fletcher_Freeman(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ILS_FF_hyper_parameters)

# logger.info("GLL线搜索下的FF方法") # 可收敛
GLL_FF_hyper_parameters = {
    "GLL": {
        "rho": 0.5,
        "sigma": 0.5,
        "M": 10,
        "a": 10**5,
    },
    "search_mode": "GLL",
    "epsilon": 1e-8,
    "max_epoch": 1000,
}

# X_star, func_X_star, iter_num = FF.Fletcher_Freeman(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=GLL_FF_hyper_parameters)

method_list = [newton_method.damp_newton, newton_method.damp_newton, newton_method.damp_newton, 
                newton_method.GM_newton, newton_method.GM_newton, newton_method.GM_newton,
                FF.Fletcher_Freeman, FF.Fletcher_Freeman, FF.Fletcher_Freeman]
method_name_list = ["精确线搜索下的阻尼牛顿法", "非精确线搜索下的阻尼牛顿法", "GLL线搜索下的阻尼牛顿法", 
                    "精确线搜索下的GM稳定牛顿法", "非精确线搜索下的GM稳定牛顿法", "GLL线搜索下的GM稳定牛顿法",
                    "精确线搜索下的FF方法", "非精确线搜索下的FF方法", "GLL线搜索下的FF方法"]
hyper_parameters_list = [ELS_damp_newton_hyper_parameters, ILS_damp_newton_hyper_parameters, GLL_damp_newton_hyper_parameters,
                    ELS_GM_newton_hyper_parameters, ILS_GM_newton_hyper_parameters, ELS_GM_newton_hyper_parameters,
                    ELS_FF_hyper_parameters, ILS_FF_hyper_parameters, ELS_FF_hyper_parameters]             

parser = argparse.ArgumentParser(description='Optimization') 
parser.add_argument("--m", type=int, default=20, help="测试函数的维度")
parser.add_argument("--test_fucntion", choices=["Wood", "EPS", "Trig"], type=str, default="EPS", help="测试函数的维度")            
args = parser.parse_args()
m = args.m
if args.test_fucntion == "EPS":
    logger.info("== " * 20 + " Extended Powell Singular Function: {} ".format(m) + "== " * 20)
    x0 = np.array([3, -1, 0, 1] * int(m / 4))
    with open("cached_expression/g_extended_powell_singular_{m}.pkl".format(m=m), 'rb') as reader:
        diff_list = pickle.load(reader)
    with open("cached_expression/G_extended_powell_singular_{m}.pkl".format(m=m), 'rb') as reader:
        G = pickle.load(reader)
    with open("cached_expression/symbols_extended_powell_singular_{m}.pkl".format(m=m), 'rb') as reader:
        symbols_list = pickle.load(reader)
    g_EPS_partial = functools.partial(functions.g_function, diff_list=diff_list, symbols_list=symbols_list)
    G_EPS_partial = functools.partial(functions.G_function, G_lists=G, symbols_list=symbols_list)

    results = []
    pool = multiprocessing.Pool(processes=len(hyper_parameters_list))
 
    for method_idx in range(len(hyper_parameters_list)):
        results.append([pool.apply_async(method_list[method_idx], (x0, functions.extended_powell_singular, g_EPS_partial, G_EPS_partial, hyper_parameters_list[method_idx], ))])
    pool.close()
    pool.join()
    # for idx, reslut in enumerate(results):
    #     logger.info(method_name_list[idx])
    #     logger.info(reslut)
    #     logger.info("迭代轮次{iter}，最终X={X}，最终函数值{value}".format(iter=reslut[0], X=reslut[1], value=reslut[2]))
    
    # logger.info("精确线搜索下的阻尼牛顿法") 
    # X_star, func_X_star, iter_num = newton_method.damp_newton(x0, functions.extended_powell_singular, g_EPS_partial, G_EPS_partial, hyper_parameters=ELS_damp_newton_hyper_parameters)
    # logger.info("非精确线搜索下的阻尼牛顿法")
    # X_star, func_X_star, iter_num = newton_method.damp_newton(x0, functions.extended_powell_singular, g_EPS_partial, G_EPS_partial, hyper_parameters=ILS_damp_newton_hyper_parameters)
    # logger.info("GLL线搜索下的阻尼牛顿法")
    # X_star, func_X_star, iter_num = newton_method.damp_newton(x0, functions.extended_powell_singular, g_EPS_partial, G_EPS_partial, hyper_parameters=GLL_damp_newton_hyper_parameters)
    # logger.info("精确线搜索下的GM稳定牛顿法") 
    # X_star, func_X_star, iter_num = newton_method.GM_newton(x0, functions.extended_powell_singular, g_EPS_partial, G_EPS_partial, hyper_parameters=ELS_GM_newton_hyper_parameters)
    # logger.info("非精确线搜索下的GM稳定牛顿法")
    # X_star, func_X_star, iter_num = newton_method.GM_newton(x0, functions.extended_powell_singular, g_EPS_partial, G_EPS_partial, hyper_parameters=ILS_GM_newton_hyper_parameters)
    # logger.info("GLL线搜索下的GM牛顿法")
    # X_star, func_X_star, iter_num = newton_method.GM_newton(x0, functions.extended_powell_singular, g_EPS_partial, G_EPS_partial, hyper_parameters=GLL_GM_newton_hyper_parameters)
    # logger.info("精确线搜索下的FF方法")
    # X_star, func_X_star, iter_num = FF.Fletcher_Freeman(x0, functions.extended_powell_singular, g_EPS_partial, G_EPS_partial, hyper_parameters=ELS_FF_hyper_parameters)
    # logger.info("非精确线搜索下的FF方法")
    # X_star, func_X_star, iter_num = FF.Fletcher_Freeman(x0, functions.extended_powell_singular, g_EPS_partial, G_EPS_partial, hyper_parameters=ILS_FF_hyper_parameters)
    # logger.info("GLL线搜索下的FF方法")
    # X_star, func_X_star, iter_num = FF.Fletcher_Freeman(x0, functions.extended_powell_singular, g_EPS_partial, G_EPS_partial, hyper_parameters=GLL_FF_hyper_parameters)

elif args.test_fucntion == "Trig":
    logger.info("== " * 20 + " Trigonometric Function: {} ".format(m) + "== " * 20)
    x0 = np.array([1/m] * int(m))
    with open("cached_expression/g_trigonometric_{m}.pkl".format(m=m), 'rb') as reader:
        diff_list = pickle.load(reader)
    with open("cached_expression/G_trigonometric_{m}.pkl".format(m=m), 'rb') as reader:
        G = pickle.load(reader)
    with open("cached_expression/symbols_trigonometric_{m}.pkl".format(m=m), 'rb') as reader:
        symbols_list = pickle.load(reader)
    g_Trig_partial = functools.partial(functions.g_function, diff_list=diff_list, symbols_list=symbols_list)
    G_Trig_partial = functools.partial(functions.G_function, G_lists=G, symbols_list=symbols_list)

    results = []
    pool = multiprocessing.Pool(processes=len(hyper_parameters_list))
 
    for method_idx in range(len(hyper_parameters_list)):
        results.append([pool.apply_async(method_list[method_idx], (x0, functions.extended_powell_singular, g_Trig_partial, G_Trig_partial, hyper_parameters_list[method_idx], ))])
    pool.close()
    pool.join()
    

    # logger.info("精确线搜索下的阻尼牛顿法") 
    # X_star, func_X_star, iter_num = newton_method.damp_newton(x0, functions.trigonometric, g_Trig_partial, G_Trig_partial, hyper_parameters=ELS_damp_newton_hyper_parameters)
    # logger.info("非精确线搜索下的阻尼牛顿法")
    # X_star, func_X_star, iter_num = newton_method.damp_newton(x0, functions.trigonometric, g_Trig_partial, G_Trig_partial, hyper_parameters=ILS_damp_newton_hyper_parameters)
    # logger.info("GLL线搜索下的阻尼牛顿法")
    # X_star, func_X_star, iter_num = newton_method.damp_newton(x0, functions.trigonometric, g_Trig_partial, G_Trig_partial, hyper_parameters=GLL_damp_newton_hyper_parameters)
    # logger.info("精确线搜索下的GM稳定牛顿法") 
    # X_star, func_X_star, iter_num = newton_method.GM_newton(x0, functions.trigonometric, g_Trig_partial, G_Trig_partial, hyper_parameters=ELS_GM_newton_hyper_parameters)
    # logger.info("非精确线搜索下的GM稳定牛顿法")
    # X_star, func_X_star, iter_num = newton_method.GM_newton(x0, functions.trigonometric, g_Trig_partial, G_Trig_partial, hyper_parameters=ILS_GM_newton_hyper_parameters)
    # logger.info("GLL线搜索下的GM牛顿法")
    # X_star, func_X_star, iter_num = newton_method.GM_newton(x0, functions.trigonometric, g_Trig_partial, G_Trig_partial, hyper_parameters=GLL_GM_newton_hyper_parameters)
    # logger.info("精确线搜索下的FF方法")
    # X_star, func_X_star, iter_num = FF.Fletcher_Freeman(x0, functions.trigonometric, g_Trig_partial, G_Trig_partial, hyper_parameters=ELS_FF_hyper_parameters)
    # logger.info("非精确线搜索下的FF方法")
    # X_star, func_X_star, iter_num = FF.Fletcher_Freeman(x0, functions.trigonometric, g_Trig_partial, G_Trig_partial, hyper_parameters=ILS_FF_hyper_parameters)
    # logger.info("GLL线搜索下的FF方法")
    # X_star, func_X_star, iter_num = FF.Fletcher_Freeman(x0, functions.trigonometric, g_Trig_partial, G_Trig_partial, hyper_parameters=GLL_FF_hyper_parameters)

else:
    logger.info("== " * 20 + " Wood Funciton " + "== " * 20)
    for method_idx in range(len(hyper_parameters_list)):
        logger.info(method_name_list[method_idx])
        X_star, func_X_star, iter_num = method_list[method_idx](x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters_list[method_idx])



    
