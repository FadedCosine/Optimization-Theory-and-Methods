import numpy as np
from goto import with_goto
import copy
import functions 
import functools

def inexact_line_search(func, gfunc, X, d, hyper_parameters=None, rho=0.1, sigma=0.4, criterion='Armijo Goldstein', start=0, end=1e10, alpha0=1e-6, t=5, symbols_list=None, appendix=False):
    """[summary]

    Args:
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        X ([np.array]]): [初值点]
        d ([np.array]]): [下降方向]
        start (int, optional): [步长下界]. Defaults to 0.
        end ([type], optional): [步长上界]. Defaults to 1e10.
        hyper_parameters: (Dic): 超参数，超参数中包括：
            rho (float, optional): [Armijo准则中的参数]. Defaults to 0.1, range in (0, 1/2).
            sigma (float, optional): [Wolfe准则中的参数]. Defaults to 0.4, range in (rho, 1).
            criterion (str, optional): [准则名称]. Defaults to 'Wolfe Powell'. 从["Armijo Goldstein", "Wolfe Powell", "Strong Wolfe Powell"]中选择
            alpha0 (float, optional): 初始步长. Defaults to 1e-6
        symbols_list ([list]): 导函数的变量符号列表
        appendix (bool, optional): [description]. Defaults to False.

    Returns:
        [float]: [搜索得到的步长]]
    """
 
    if hyper_parameters is not None:
        rho = hyper_parameters["rho"]
        sigma = hyper_parameters["sigma"]
        criterion = hyper_parameters["criterion"]
        alpha = hyper_parameters["alpha0"]
    else:
        alpha = alpha0
    # if appendix == True:
    #     alpha0 = (start + end) / 2   # save initial point
    
    # reduce unnecessary caculations in loop
    f0, gf0 = func(X), gfunc(X)
    # gf0 must be a numpy array
    gkdk = gf0.dot(d)
    wolfe_boundary = sigma * gkdk
    strong_wolfe_boundary = sigma * abs(gkdk)

    iter_num = 0
    while True:
        armijo_boundary = f0 + rho * gkdk * alpha
        goldstein_boundary = f0 + (1 - rho) * gkdk * alpha
        fAlpha, gfAlpha = func(X + alpha * d), gfunc(X + alpha * d)
        gkAlpha_dk = gfAlpha.dot(d)
        # different criterions have same condition1 to avoid too large alpha
        armijo_condition = (fAlpha <= armijo_boundary)
        # different criterions have different condition2 to avoid too small alpha
        if criterion == 'Armijo Goldstein':
            condition2 = (fAlpha >= goldstein_boundary)
        elif criterion == 'Wolfe Powell':
            condition2 = (gkAlpha_dk >= wolfe_boundary)
        elif criterion == 'Strong Wolfe Powell':
            condition2 = (abs(gkAlpha_dk) <= strong_wolfe_boundary)
        else:
            condition2 = True

        # update start or end point or stop iteration
        if armijo_condition == False:
            end = alpha
            alpha = (start + end) / 2
        elif condition2 == False:
        # elif alpha < minstep:
            start = alpha
            if end < 1e10:
                alpha = (start + end) / 2
            else:
                alpha = t * alpha
        else:
            alpha_star = alpha
            min_value = fAlpha
            break
        iter_num += 1

    if appendix == True:
        print("方法：非精确线搜索；准则：%s\n" % criterion)
        print("初始步长：%.2f" % (alpha0))
        print("初始点函数值：%.2f" % (f0))
        print("停止步长：%.4f; 停止点函数值：%.4f; 迭代次数：%d" % (alpha_star, min_value, iter_num))

    return alpha_star

def test():
    x0 = np.array([-3, -1, -3, -1])
    d0 = np.array([2, 1, 2, 1])
    diff_wood_list, symbols_wood_list = functions.diff_wood_expression()
    g_wood_partial = functools.partial(functions.g_wood, diff_list=diff_wood_list, symbols_list=symbols_wood_list)
    alpha_star = inexact_line_search(functions.wood, g_wood_partial, x0, d0, appendix=True)
    print(functions.wood(x0 + d0 * alpha_star))

def main():
    test()

if __name__ == "__main__":
    main()
