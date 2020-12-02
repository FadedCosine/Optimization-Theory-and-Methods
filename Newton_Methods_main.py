import functions
import Newton_Methods.fletcher_freeman as FF
import Newton_Methods.newton_method as newton_method
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
CRITERION = ["Armijo Goldstein", "Wolfe Powell", "Strong Wolfe Powell"]
ILS_criterion = CRITERION[0]
# logger.info("精确线搜索下的阻尼牛顿法") # 使用基本牛顿法的下降方向无法收敛，使用修正Cholesky分解的下降方向可以收敛
ELS_damp_newton_hyper_parameters = {
    "ELS": {
        "retreat_method": {
            "a0" : 0,
            "r": 1e-8,
            "t": 1.5,
        },
        "golden_method": {
            "epsilon": 1e-6,
        }
    },
    "damp_newton": {
        "use_modified_Cholesky" : False,
    },
    "modified_Cholesky": {
        "u": 1e-20,
    },
    "search_mode": "ELS",
    "epsilon": 1e-8,
    "max_epoch": 1000,
}
# X_star, func_X_star, iter_num = newton_method.damp_newton(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ELS_damp_newton_hyper_parameters)


# logger.info("非精确线搜索下的阻尼牛顿法") # 可收敛
ILS_damp_newton_hyper_parameters = {
    "ILS": {
        "rho": 0.2,
        "sigma": 0.5,
        "t": 5,
        "alpha0": 1e-8,
        "criterion": ILS_criterion
    },
    "damp_newton": {
        "use_modified_Cholesky" : False,
    },
    "modified_Cholesky": {
            "u": 1e-20,
    },
    "search_mode": "ILS",
    "epsilon": 1e-8,
    "max_epoch": 1000,
}
# X_star, func_X_star, iter_num = newton_method.damp_newton(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ILS_damp_newton_hyper_parameters)

# logger.info("GLL线搜索下的阻尼牛顿法") # 可收敛
GLL_damp_newton_hyper_parameters = {
    "GLL": {
        "rho": 0.25,
        "sigma": 0.5,
        "M": 15,
        "a": 1,
    },
    "damp_newton": {
        "use_modified_Cholesky" : False,
    },
    "modified_Cholesky": {
        "u": 1e-20,
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
            "a0" : 0,
            "r": 1e-10,
            "t": 1.5,
        },
        "golden_method": {
            "epsilon": 1e-7,
        }
    },
    "GM_newton": {
        "zeta": 1e-8,
    },
    "modified_Cholesky": {
        "u": 1e-20,
    },
    "search_mode": "ELS",
    "epsilon": 1e-8,
    "max_epoch": 1000,
}
# X_star, func_X_star, iter_num = newton_method.GM_newton(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ELS_GM_newton_hyper_parameters)

# logger.info("非精确线搜索下的GM稳定牛顿法") # 可收敛
ILS_GM_newton_hyper_parameters = {
    "ILS": {
        "rho": 0.2,
        "sigma": 0.5,
        "t": 5,
        "alpha0": 1e-8,
        "criterion": ILS_criterion
    },
    "GM_newton": {
        "zeta": 1e-8,
    },
    "modified_Cholesky": {
        "u": 1e-20,
    },
    "search_mode": "ILS",
    "epsilon": 1e-8,
    "max_epoch": 1000,
}
# X_star, func_X_star, iter_num = newton_method.GM_newton(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ILS_GM_newton_hyper_parameters)

# logger.info("GLL线搜索下的GM牛顿法") # 可收敛

GLL_GM_newton_hyper_parameters = {
    "GLL": {
        "rho": 0.25,
        "sigma": 0.5,
        "M": 15,
        "a": 1,
    },
    "GM_newton": {
        "zeta": 1e-8,
    },
    "modified_Cholesky": {
        "u": 1e-20,
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
            "a0" : 0, 
            "r": 1e-10,
            "t": 1.5,
        },
        "golden_method": {
            "epsilon": 1e-7,
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
        "criterion": ILS_criterion
    },
    "search_mode": "ILS",
    "epsilon": 1e-8,
    "max_epoch": 1000,
}

# X_star, func_X_star, iter_num = FF.Fletcher_Freeman(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ILS_FF_hyper_parameters)

# logger.info("GLL线搜索下的FF方法") # 可收敛
GLL_FF_hyper_parameters = {
    "GLL": {
        "rho": 0.25,
        "sigma": 0.5,
        "M": 12,
        "a": 1,
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
                    ELS_GM_newton_hyper_parameters, ILS_GM_newton_hyper_parameters, GLL_GM_newton_hyper_parameters,
                    ELS_FF_hyper_parameters, ILS_FF_hyper_parameters, GLL_FF_hyper_parameters]             

parser = argparse.ArgumentParser(description='Optimization') 
parser.add_argument("--m", type=int, default=20, help="测试函数的维度")
parser.add_argument("--test_fucntion", choices=["Wood", "EPS", "Trig"], type=str, default="EPS", help="测试函数的维度")            
args = parser.parse_args()
m = args.m
if args.test_fucntion == "EPS":
    x0 = np.array([3, -1, 0, 1] * int(m//4))
    f_funciton = functions.extended_powell_singular
    g_function = functions.g_EPS
    G_function = functions.G_EPS
    write_latex_name = "EPS_{}.txt".format(m)
elif args.test_fucntion == "Trig":
    x0 = np.array([1/m] * int(m))
    f_funciton = functions.trigonometric
    g_function = functions.g_trigonometric
    G_function = functions.G_trigonometric
    write_latex_name = "Trig_{}.txt".format(m)
else:
    x0 = np.array([-3, -1, -3, -1])
    f_funciton = functions.wood
    diff_wood_list, symbols_wood_list = functions.diff_wood_expression()
    g_function = functools.partial(functions.g_wood, diff_list=diff_wood_list, symbols_list=symbols_wood_list)
    hess_wood_lists, symbols_wood_list = functions.hess_wood_expression()
    G_function = functools.partial(functions.G_wood, G_lists=hess_wood_lists, symbols_list=symbols_wood_list)
    write_latex_name = "Wood.txt"


   
results = []
pool = multiprocessing.Pool(processes=len(hyper_parameters_list))

for method_idx in range(len(hyper_parameters_list)):
    results.append([pool.apply_async(method_list[method_idx], (x0, f_funciton, g_function, G_function, hyper_parameters_list[method_idx], ))])
pool.close()
pool.join()
logger.info("== " * 20 + " {} ".format(write_latex_name) + "== " * 20)
write_latex = open(write_latex_name, 'w')
write_latex.write("\hline\n")
logger.info("精确线搜索下的阻尼牛顿法") 
X_star, func_X_star, iter_num, function_num = newton_method.damp_newton(x0, f_funciton, g_function, G_function, hyper_parameters=ELS_damp_newton_hyper_parameters)
write_latex.write(" 阻尼牛顿法 & ELS & {fx} & {iter_num} & {func_k} & {is_conv} \\\\ \n".format(
    fx = format(func_X_star, ".4e"),
    iter_num = str(iter_num),
    func_k = str(function_num),
    is_conv = "是" if func_X_star < 1e-5 else "否"
))
logger.info("非精确线搜索下的阻尼牛顿法")
X_star, func_X_star, iter_num, function_num = newton_method.damp_newton(x0, f_funciton, g_function, G_function, hyper_parameters=ILS_damp_newton_hyper_parameters)
write_latex.write(" 阻尼牛顿法 & ILS & {fx} & {iter_num} & {func_k} & {is_conv} \\\\ \n".format(
    fx = format(func_X_star, ".4e"),
    iter_num = str(iter_num),
    func_k = str(function_num),
    is_conv = "是" if func_X_star < 1e-5 else "否"
))
logger.info("GLL线搜索下的阻尼牛顿法")
X_star, func_X_star, iter_num, function_num = newton_method.damp_newton(x0, f_funciton, g_function, G_function, hyper_parameters=GLL_damp_newton_hyper_parameters)
write_latex.write(" 阻尼牛顿法 & GLL & {fx} & {iter_num} & {func_k} & {is_conv} \\\\ \n".format(
    fx = format(func_X_star, ".4e"),
    iter_num = str(iter_num),
    func_k = str(function_num),
    is_conv = "是" if func_X_star < 1e-5 else "否"
))
write_latex.write("\hline\n")
logger.info("精确线搜索下的GM稳定牛顿法") 
X_star, func_X_star, iter_num, function_num = newton_method.GM_newton(x0, f_funciton, g_function, G_function, hyper_parameters=ELS_GM_newton_hyper_parameters)
write_latex.write(" GM稳定牛顿法 & ELS & {fx} & {iter_num} & {func_k} & {is_conv} \\\\ \n".format(
    fx = format(func_X_star, ".4e"),
    iter_num = str(iter_num),
    func_k = str(function_num),
    is_conv = "是" if func_X_star < 1e-5 else "否"
))
logger.info("非精确线搜索下的GM稳定牛顿法")
X_star, func_X_star, iter_num, function_num = newton_method.GM_newton(x0, f_funciton, g_function, G_function, hyper_parameters=ILS_GM_newton_hyper_parameters)
write_latex.write(" GM稳定牛顿法 & ILS & {fx} & {iter_num} & {func_k} & {is_conv} \\\\ \n".format(
    fx = format(func_X_star, ".4e"),
    iter_num = str(iter_num),
    func_k = str(function_num),
    is_conv = "是" if func_X_star < 1e-5 else "否"
))
logger.info("GLL线搜索下的GM牛顿法")
X_star, func_X_star, iter_num, function_num = newton_method.GM_newton(x0, f_funciton, g_function, G_function, hyper_parameters=GLL_GM_newton_hyper_parameters)
write_latex.write(" GM稳定牛顿法 & GLL & {fx} & {iter_num} & {func_k} & {is_conv} \\\\ \n".format(
    fx = format(func_X_star, ".4e"),
    iter_num = str(iter_num),
    func_k = str(function_num),
    is_conv = "是" if func_X_star < 1e-5 else "否"
))
write_latex.write("\hline\n")
logger.info("精确线搜索下的FF方法")
X_star, func_X_star, iter_num, function_num = FF.Fletcher_Freeman(x0, f_funciton, g_function, G_function, hyper_parameters=ELS_FF_hyper_parameters)
write_latex.write(" Fletcher-Freeman 方法 & ELS & {fx} & {iter_num} & {func_k} & {is_conv} \\\\ \n".format(
    fx = format(func_X_star, ".4e"),
    iter_num = str(iter_num),
    func_k = str(function_num),
    is_conv = "是" if func_X_star < 1e-5 else "否"
))
logger.info("非精确线搜索下的FF方法")
X_star, func_X_star, iter_num, function_num = FF.Fletcher_Freeman(x0, f_funciton, g_function, G_function, hyper_parameters=ILS_FF_hyper_parameters)
write_latex.write(" Fletcher-Freeman 方法 & ILS & {fx} & {iter_num} & {func_k} & {is_conv} \\\\ \n".format(
    fx = format(func_X_star, ".4e"),
    iter_num = str(iter_num),
    func_k = str(function_num),
    is_conv = "是" if func_X_star < 1e-5 else "否"
))
logger.info("GLL线搜索下的FF方法")
X_star, func_X_star, iter_num, function_num = FF.Fletcher_Freeman(x0, f_funciton, g_function, G_function, hyper_parameters=GLL_FF_hyper_parameters)
write_latex.write(" Fletcher-Freeman 方法 & GLL & {fx} & {iter_num} & {func_k} & {is_conv} \\\\ \n".format(
    fx = format(func_X_star, ".4e"),
    iter_num = str(iter_num),
    func_k = str(function_num),
    is_conv = "是" if func_X_star < 1e-5 else "否"
))
write_latex.write("\hline\n")
write_latex.close()



