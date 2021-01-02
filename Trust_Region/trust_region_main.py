import functions
import numpy as np
import math
import time
from goto import with_goto
import utils
import functools
from scipy.sparse.linalg import gmres
from Trust_Region.hebden import hebden
from Trust_Region.sorensen import sorensen
from Trust_Region.two_subspace_min import two_subspace_min
import argparse
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

@with_goto
def trust_region_method(X, func, gfunc, hess_func, hyper_parameters=None, TR_method=hebden, delta=0.1, epsilon=1e-9, max_epoch=1000):
    
    """[ 信赖域算法的主要框架，可选择不同的子问题求解方法

    Args:
        X ([np.array]): [Input X]
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        hess_func ([回调函数]): [目标函数的Hessian矩阵]
        hyper_parameters: (Dic): 超参数，超参数中包括：
            TR_method (str, optional): [子问题的求解方法]. Defaults to 'Sorensen'. ['Hebden', 'Sorensen', '二维子空间极小化方法']
            epsilon ([float], optional): [||g_k|| < 1e-5 * max(1, ||x_k||)时，迭代结束]. Defaults to 1e-8.
            max_epoch (int, optional): [最大允许的迭代次数]. Defaults to 1000.
    
    Returns:
        返回求解得到的极小值点，极小值点对应的函数值和迭代次数
    """
    if hyper_parameters is not None:
        TR_method = hyper_parameters["TR"]["TR_method"]
        delta = hyper_parameters["TR"]["delta"]
        epsilon = hyper_parameters["epsilon"]
        max_epoch = hyper_parameters["max_epoch"]
    n = len(X)
    k = 0
    TR_iter_k = 0
    function_k = 0
    start_time = time.time()

    label.step2
    function_k += 1
    F = func(X)
    g = gfunc(X)
    G = hess_func(X)
    
    # if np.linalg.norm(g) < epsilon:
        
    #     logger.info("因为满足终止条件，{mode}方法，迭代结束，迭代轮次{iter}，函数调用次数{func_k}，最终用时{time}，最终X={X}，最终函数值={func}".format(mode=TR_method.__name__, iter=k, func_k=function_k, time=end_time-start_time, X=X,func=F))
    #     return X, F, k, function_k, TR_iter_k, end_time-start_time

    d, add_iter_k = TR_method(X, func, gfunc, hess_func, delta)
    TR_iter_k += add_iter_k
    end_time = time.time()
    function_k += 1
    logger.info("迭代第{iter}轮，当前函数调用次数{func_k}，求解TR子问题共迭代次数{TR_k}，当前用时{time}，当前X取值为{X}，当前g的取值为{g}, 下降方向为{d}，当前函数值为{func_x}".format(iter=k,func_k=function_k,TR_k=TR_iter_k, time=end_time-start_time,X=X, g=g, d=d,func_x=round(F, 8)))

    X_tmp = X + d
    F_tmp = func(X_tmp)
    if abs(F - F_tmp) < epsilon:
        end_time = time.time()
        logger.info("因为满足终止条件，{mode}方法，迭代结束，迭代轮次{iter}，函数调用次数{func_k}，求解TR子问题共迭代次数{TR_k}，最终用时{time}，最终X={X}，最终函数值={func}".format(mode=TR_method.__name__, iter=k, func_k=function_k,TR_k=TR_iter_k, time=end_time-start_time, X=X,func=F))
        return X, F, k, function_k, TR_iter_k, end_time-start_time


    q_k = -(g @ d + 0.5 * d @ G @ d)
    gamma_k = (F - F_tmp) / q_k
    # logger.info("F - F_tmp is {}".format(F - F_tmp))
    # logger.info("q_k is {}".format(q_k))
    # logger.info("gamma_k is {}".format(gamma_k))

    if gamma_k >= 0.75 and abs(np.linalg.norm(d) - delta) <= epsilon:
        delta = delta * 2
    elif gamma_k <= 0.25:
        delta = delta / 4
    
    if gamma_k > 0:
        X = X_tmp
    k = k + 1
    goto.step2

if __name__ == '__main__':
    
        
    parser = argparse.ArgumentParser(description='Optimization') 
    parser.add_argument("--m", type=int, default=100, help="测试函数的维度")
    parser.add_argument("--delta", type=float, default=0.5, help="信赖域问题中delta的取值")
    parser.add_argument("--test_fucntion", choices=["Wood", "EPS", "Trig", "Penalty1", "EFR", "ER"], type=str, default="EPS", help="测试函数的维度")            
    args = parser.parse_args()
    m = args.m
    delta = args.delta
    Hebden_hyper_parameters = {
        "TR":{
            "TR_method": hebden,
            "delta": delta,
        },
        "epsilon": 1e-8,
        "max_epoch": 1000,
    }
    Sorensen_hyper_parameters = {
        "TR":{
            "TR_method": sorensen,
            "delta": delta,
        },
        "epsilon": 1e-8,
        "max_epoch": 1000,
    }
    TSM_hyper_parameters = {
        "TR":{
            "TR_method": two_subspace_min,
            "delta": delta,
        },
        "epsilon": 1e-8,
        "max_epoch": 1000,
    }
    
    if args.test_fucntion == "EPS":
        X = np.array([3, -1, 0, 1] * int(m//4))
        test_function = functions.EPS(m)
        f_funciton = test_function.func
        g_function = test_function.gfunc
        G_function = test_function.hess_func
        write_latex_name = "EPS_{}_delta{}.txt".format(m,delta)

    elif args.test_fucntion == "Trig":
        X = np.array([1/m] * int(m))
        test_function = functions.Trigonometric(m)
        f_funciton = test_function.func
        g_function = test_function.gfunc
        G_function = test_function.hess_func
        write_latex_name = "Trig_{}_delta{}.txt".format(m,delta)

    elif args.test_fucntion == "Trig":
        X = np.array([-3, -1, -3, -1])
        f_funciton = functions.wood
        diff_wood_list, symbols_wood_list = functions.diff_wood_expression()
        g_function = functools.partial(functions.g_wood, diff_list=diff_wood_list, symbols_list=symbols_wood_list)
        hess_wood_lists, symbols_wood_list = functions.hess_wood_expression()
        G_function = functools.partial(functions.G_wood, G_lists=hess_wood_lists, symbols_list=symbols_wood_list)
        write_latex_name = "Wood_delta{}.txt".format(delta)
    elif args.test_fucntion == "Penalty1":
        X = np.array(range(1, m + 1))
        test_function = functions.Penalty1(m)
        f_funciton = test_function.func
        g_function = test_function.gfunc
        G_function = test_function.hess_func
        write_latex_name = "Penalty1_{}_delta{}.txt".format(m,delta)
    elif args.test_fucntion == "EFR":
        X = np.array([-2.] * m)
        test_function = functions.Extended_Freudenstein_Roth(m)
        f_funciton = test_function.func
        g_function = test_function.gfunc
        G_function = test_function.hess_func
        write_latex_name = "EFR_{}_delta{}.txt".format(m,delta)
    elif args.test_fucntion == "ER":
        test_function = functions.Extended_Rosenbrock(m)
        X = np.zeros(m)
        t = np.array(range(int(m / 2)))
        X[2 * t] = -1.2
        X[2 * t + 1] = 1
        f_funciton = test_function.func
        g_function = test_function.gfunc
        G_function = test_function.hess_func
        write_latex_name = "ER_{}_delta{}.txt".format(m,delta)


    logger.info("== " * 20 + " {} ".format(write_latex_name) + "== " * 20)
    write_latex = open(write_latex_name, 'w')

    logger.info("Hebden方法") 
    X_star, func_X_star, iter_num, function_num, TR_iter_num, cpu_time= trust_region_method(X, f_funciton, g_function, G_function, hyper_parameters=Hebden_hyper_parameters)
    write_latex.write(" Hebden & {fx} & {iter_num} & {func_k} & {TR_k} & {cpu_time} & {is_conv} \\\\ \n".format(
        fx = format(func_X_star, ".4e"),
        iter_num = str(iter_num),
        func_k = str(function_num),
        TR_k = str(TR_iter_num),
        cpu_time = round(cpu_time, 4),
        is_conv = "是" if func_X_star < 1e-5 else "否"
    ))

    logger.info("More-Sorensen方法") 
    X_star, func_X_star, iter_num, function_num, TR_iter_num, cpu_time= trust_region_method(X, f_funciton, g_function, G_function, hyper_parameters=Sorensen_hyper_parameters)

    write_latex.write(" More-Sorensen & {fx} & {iter_num} & {func_k} & {TR_k} & {cpu_time} & {is_conv} \\\\ \n".format(
        fx = format(func_X_star, ".4e"),
        iter_num = str(iter_num),
        func_k = str(function_num),
        TR_k = str(TR_iter_num),
        cpu_time = round(cpu_time, 4),
        is_conv = "是" if func_X_star < 1e-5 else "否"
    ))

    logger.info("二维子空间极小化") 
    X_star, func_X_star, iter_num, function_num, TR_iter_num, cpu_time= trust_region_method(X, f_funciton, g_function, G_function, hyper_parameters=TSM_hyper_parameters)
  
    write_latex.write(" 二维子空间极小化 & {fx} & {iter_num} & {func_k} & {TR_k} & {cpu_time} & {is_conv} \\\\ \n".format(
        fx = format(func_X_star, ".4e"),
        iter_num = str(iter_num),
        func_k = str(function_num),
        TR_k = str(TR_iter_num),
        cpu_time = round(cpu_time, 4),
        is_conv = "是" if func_X_star < 1e-5 else "否"
    ))


