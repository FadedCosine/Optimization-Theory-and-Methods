import functions
import exact_line_search as ELS
import inexact_line_search as ILS
import fletcher_freeman as FF
import newton_method 
import utils
import numpy as np
import functools
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
    "GM_newton": {
        "zeta": 1e-2,
    },
    "modified_Cholesky": {
        "u": 1e-10,
    },
    "search_mode": "ELS",
    "epsilon": 1e-5,
    "max_epoch": 1000,
}

x0 = np.array([-3, -1, -3, -1])
d0 = np.array([2, 1, 2, 1])
diff_wood_list, symbols_wood_list = functions.diff_wood_expression()
g_wood_partial = functools.partial(functions.g_wood, diff_list=diff_wood_list, symbols_list=symbols_wood_list)
hess_wood_lists, symbols_wood_list = functions.hess_wood_expression()
G_wood_partial = functools.partial(functions.G_wood, G_lists=hess_wood_lists, symbols_list=symbols_wood_list)

# print("精确线搜索下的阻尼牛顿法") # 无法收敛
# ELS_damp_newton_hyper_parameters = {
#     "ELS": {
#         "retreat_method": {
#             "a0" : 1e-2 ,
#             "r": 1e-6,
#             "t": 5,
#         },
#         "golden_method": {
#             "epsilon": 1e-6,
#         }
#     },
#     "modified_Cholesky": {
#         "u": 1e-10,
#     },
#     "search_mode": "ELS",
#     "epsilon": 1e-5,
#     "max_epoch": 1000,
# }
# X_star, func_X_star, iter_num = newton_method.damp_newton(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ELS_damp_newton_hyper_parameters)


# print("非精确线搜索下的阻尼牛顿法") # 可收敛
# ILS_damp_newton_hyper_parameters = {
#     "ILS": {
#         "rho": 0.1,
#         "sigma": 0.4,
#         "t": 5,
#         "alpha0": 1e-6,
#         "criterion": "Armijo Goldstein"
#     },
#     "modified_Cholesky": {
#         "u": 1e-10,
#     },
#     "search_mode": "ILS",
#     "epsilon": 1e-5,
#     "max_epoch": 1000,
# }
# X_star, func_X_star, iter_num = newton_method.damp_newton(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ILS_damp_newton_hyper_parameters)

# print("精确线搜索下的GM稳定牛顿法") # 可收敛
# ELS_GM_newton_hyper_parameters = {
#     "ELS": {
#         "retreat_method": {
#             "r": 1e-5,
#             "t": 1.5,
#         },
#         "golden_method": {
#             "epsilon": 1e-5,
#         }
#     },
#     "GM_newton": {
#         "zeta": 1e-2,
#     },
#     "modified_Cholesky": {
#         "u": 1e-10,
#     },
#     "search_mode": "ELS",
#     "epsilon": 1e-5,
#     "max_epoch": 1000,
# }
# X_star, func_X_star, iter_num = newton_method.GM_newton(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ELS_GM_newton_hyper_parameters)

# print("非精确线搜索下的GM稳定牛顿法") # 可收敛
# ILS_GM_newton_hyper_parameters = {
#     "ILS": {
#         "rho": 0.1,
#         "sigma": 0.4,
#         "t": 5,
#         "alpha0": 1e-6,
#         "criterion": "Wolfe Powell"
#     },
#     "GM_newton": {
#         "zeta": 1e-2,
#     },
#     "modified_Cholesky": {
#         "u": 1e-10,
#     },
#     "search_mode": "ILS",
#     "epsilon": 1e-5,
#     "max_epoch": 1000,
# }
# X_star, func_X_star, iter_num = newton_method.GM_newton(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ILS_GM_newton_hyper_parameters)

print("精确线搜索下的FF方法") # 可收敛
ELS_FF_hyper_parameters = {
    "ELS": {
        "retreat_method": {
            "a0": 1e-2, 
            "r": 1e-4,
            "t": 10,
        },
        "golden_method": {
            "epsilon": 1e-5,
        }
    },
    "search_mode": "ELS",
    "epsilon": 1e-6,
    "max_epoch": 10000,
}
FF.Fletcher_Freeman(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ELS_FF_hyper_parameters)

print("非精确线搜索下的FF方法") # 可收敛
ILS_FF_hyper_parameters = {
    "ILS": {
        "rho": 0.1,
        "sigma": 0.4,
        "t": 5,
        "alpha0": 1e-6,
        "criterion": "Armijo Goldstein"
    },
    "search_mode": "ILS",
    "epsilon": 1e-5,
    "max_epoch": 1000,
}

FF.Fletcher_Freeman(x0, functions.wood, g_wood_partial, G_wood_partial, hyper_parameters=ILS_FF_hyper_parameters)