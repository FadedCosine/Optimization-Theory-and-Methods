import functions
import numpy as np
from goto import with_goto
import exact_line_search as ELS
import inexact_line_search as ILS
import utils
import functools
import copy

@with_goto
def damp_newton(X, func, gfunc, hess_funct, search_mode='ELS', ILS_model="Wolfe Powell", epsilon=1e-5, max_epoch=1000):
    """[使用阻尼牛顿法极小值点
         d = -G_k^{-1} * g_k]

    Args:
        X ([np.array]): [Input X]
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        hess_funct ([回调函数]): [目标函数的Hessian矩阵]
        search_mode (str, optional): [线搜索的模式（选择精确线搜索还是非精确线搜索）]. Defaults to 'ELS'. ['ELS', 'ILS']
        ILS_model (str, optional): [如果是非精确线搜索，选择非精确线搜索采用的准则]. Defaults to "Wolfe Powell".
        epsilon ([float], optional): [当函数值下降小于epsilon，迭代结束]. Defaults to 1e-5.
        max_epoch (int, optional): [最大允许的迭代次数]. Defaults to 1000.

    Returns:
        返回求解得到的极小值点，极小值点对应的函数值和迭代次数
    """
    k = 1
    #计算下降方向d_k
    label .count_dk
    try:
        inv_hass = np.linalg.inv(hess_funct(X))
        d = -np.dot(inv_hass , gfunc(X))
    except:
        print("Hessian 矩阵不可逆，尝试其他方法求下降方向")
        # TODO: 加上修正Cholesky计算下降方向的参数
        d = modified_Cholesky_for_d()
    
    #计算步长
    if search_mode == "ELS":
        print("迭代第{iter}轮，当前X取值为{X}，下降方向为{d}，当前函数值为{func_x}".format(iter=k,X=X,d=d,func_x=round(func(X), 5)))
        [a, b] = ELS.retreat_method(func, X, d) 
        alpha_star = ELS.golden_method(func, X, d, a, b)
    elif search_mode == "ILS":
        print("迭代第{iter}轮，当前X取值为{X}，下降方向为{d}，当前函数值为{func_x}".format(iter=k,X=X,d=d,func_x=round(func(X), 5)))
        alpha_star = ILS.inexact_line_search(func, gfunc, X, d, criterion=ILS_model)
    else:
        raise ValueError("参数search_mode 必须从['ELS', 'ILS']当中选择")
    
    X_new = X + d * alpha_star
    func_X_new = func(X_new)
    if abs(func_X_new - func(X)) <= epsilon:
        print("迭代结束，迭代轮次{iter}，最终X={X}，最终函数值={func_X_new}".format(iter=k,X=X,func_X_new=func_X_new))
        return X_new, func_X_new, k
    if k > max_epoch:
        raise Exception("超过最大迭代次数：%d", max_epoch)
    X = X_new
    k += 1
    goto .count_dk



# TODO：实现修正Cholesky计算下降方向
def modified_Cholesky_for_d():
    return

def negative_curvature(LT, D, E):
    """计算负曲率方向

    Args:
        LT ([np.array]]): LT矩阵
        D ([np.array]): D 对角矩阵
        E ([np.array]): E 对角矩阵， 由 modified_G - G 得到

    Returns:
        [np.array]: [输出负曲率方向]，可能不存在，输出None
    """
    n = len(D)
    # 步1
    psi = np.zeros(n)
    for i in range(n):
        psi[i] = D[i][i] - E[i][i]
    # 步2
    
    t = np.where(psi==np.min(psi))[0]
    # 步3
    if psi[t] >= 0:
        return None
    else:
        pt = np.zeros(n)
        pt[t] = 1
        LT_ = np.linalg.inv(LT)  
        d = np.dot(pt, LT_) 
        return d

@with_goto
def GM_newton(X, func, gfunc, hess_funct, search_mode='ELS', ILS_model="Wolfe Powell", zeta=1e-2, epsilon=1e-5, max_epoch=1000):
    """使用Gill Murray稳定牛顿法求极小值点
         d = -G_k^{-1} * g_k]

    Args:
        X ([np.array]): [Input X]
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        hess_funct ([回调函数]): [目标函数的Hessian矩阵]
        search_mode (str, optional): [线搜索的模式（选择精确线搜索还是非精确线搜索）]. Defaults to 'ELS'. ['ELS', 'ILS']
        ILS_model (str, optional): [如果是非精确线搜索，选择非精确线搜索采用的准则]. Defaults to "Wolfe Powell".
        zeta ([float], optional): [当gk的模大于zeta， 求解方程得到下降方向]. Defaults to 1e-2.
        epsilon ([float], optional): [当函数值下降小于epsilon，迭代结束]. Defaults to 1e-5.
        max_epoch (int, optional): [最大允许的迭代次数]. Defaults to 1000.

    Returns:
        返回求解得到的极小值点，极小值点对应的函数值和迭代次数
    """
    k = 1
    assert epsilon > 0 , "must have epsilon > 0" 
    # 步2：计算g和G
    label .step2
    g = gfunc(X)
    G = hess_funct(X)
    # 步3：对G进行修正Cholesky分解
    L, D = utils.modified_Cholesky(G)
    modified_G = utils.get_modified_G(L, D)
    # 步4， ||g(x)|| > zeta ，解方程计算下降方向
    if np.linalg.norm(g) > zeta:
        G_1 = np.linalg.inv(modified_G)
        d = -np.dot(G_1, g)
        goto.step6
    # 步5：计算负曲率方向，如果psi>=0则停止，否则求出方向d
    LT = copy.deepcopy(L).T
    E = modified_G - G
    d = negative_curvature(LT, D, E)
    if d == None:
        print("因为负曲率方向不存在，迭代结束，迭代轮次{iter}，最终X={X}，最终函数值={func_X_new}".format(iter=k,X=X,func_X_new=func_X_new))
        return X, func(X), k
    else:
        gT = np.mat(g).T
        if np.dot(gT, d) > 0:
            d = -d
    # 步6：线搜索求步长
    label .step6
    if search_mode == "ELS":
        print("迭代第{iter}轮，当前X取值为{X}，下降方向为{d}，当前函数值为{func_x}".format(iter=k,X=X,d=d,func_x=round(func(X), 5)))
        [a, b] = ELS.retreat_method(func, X, d) 
        alpha_star = ELS.golden_method(func, X, d, a, b)
    elif search_mode == "ILS":
        print("迭代第{iter}轮，当前X取值为{X}，下降方向为{d}，当前函数值为{func_x}".format(iter=k,X=X,d=d,func_x=round(func(X), 5)))
        alpha_star = ILS.inexact_line_search(func, gfunc, X, d, criterion=ILS_model)
    else:
        raise ValueError("参数search_mode 必须从['ELS', 'ILS']当中选择")

    X_new = X + d * alpha_star
    func_X_new = func(X_new)
    if abs(func_X_new - func(X)) <= epsilon:
        print("因为函数值下降在{epsilon}以内，迭代结束，迭代轮次{iter}，最终X={X}，最终函数值={func_X_new}".format(epsilon=epsilon, iter=k,X=X,func_X_new=func_X_new))
        return X_new, func_X_new, k
    if k > max_epoch:
        raise Exception("超过最大迭代次数：%d", max_epoch)
    X = X_new
    k += 1
    goto .step2


if __name__ == '__main__':
    x0 = np.array([-3, -1, -3, -1])
    d0 = np.array([2, 1, 2, 1])
    diff_wood_list, symbols_wood_list = functions.diff_wood_expression()
    g_wood_partial = functools.partial(functions.g_wood, diff_list=diff_wood_list, symbols_list=symbols_wood_list)
    hess_wood_lists, symbols_wood_list = functions.hess_wood_expression()
    G_wood_partial = functools.partial(functions.G_wood, G_lists=hess_wood_lists, symbols_list=symbols_wood_list)
    print("精确线搜索下的阻尼牛顿法")
    X_star, func_X_star, iter_num = damp_newton(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='ELS')
    print("非精确线搜索下的阻尼牛顿法")
    X_star, func_X_star, iter_num = damp_newton(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='ILS')
    print("精确线搜索下的GM稳定牛顿法")
    X_star, func_X_star, iter_num = GM_newton(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='ELS')
    print("非精确线搜索下的GM稳定牛顿法")
    X_star, func_X_star, iter_num = GM_newton(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='ILS')
 

