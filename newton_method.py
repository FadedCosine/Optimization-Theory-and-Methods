import functions
import numpy as np
from goto import with_goto
import exact_line_search as ELS
import inexact_line_search as ILS
import functools

@with_goto
def damp_newton(X, func, gfunc, hess_funct, search_mode='ELS', ILS_model="Wolfe Powell", epsilon=1e-5, max_epoch=1000):
    """[使用阻尼牛顿法极小值点
         d = -G_k^{-1} * g_k]

    Args:
        X ([np.array]): [Input X]
        gfunc ([type]): [目标函数的一阶导函数]
        hess_funct ([type]): [目标函数的Hessian矩阵]
        
    """
    k = 0
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
    
    k += 1
    X_new = X + d * alpha_star
    func_X_new = func(X_new)
    if abs(func_X_new - func(X)) <= epsilon:
        print("迭代结束，迭代轮次{iter}，最终X={X}，最终函数值={func_X_new}".format(iter=k,X=X,func_X_new=func_X_new))
        return X_new, func_X_new, k
    if k > max_epoch:
        raise Exception("超过最大迭代次数：%d", max_epoch)
    X = X_new
    goto .count_dk



# TODO：实现修正Cholesky计算下降方向
def modified_Cholesky_for_d():
    return
    
if __name__ == '__main__':
    x0 = np.array([-3, -1, -3, -1])
    d0 = np.array([2, 1, 2, 1])
    diff_wood_list, symbols_wood_list = functions.diff_wood_expression()
    g_wood_partial = functools.partial(functions.g_wood, diff_list=diff_wood_list, symbols_list=symbols_wood_list)
    hess_wood_lists, symbols_wood_list = functions.hess_wood_expression()
    G_wood_partial = functools.partial(functions.G_wood, G_lists=hess_wood_lists, symbols_list=symbols_wood_list)
    X_star, func_X_star, iter_num = damp_newton(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='ELS')
    X_star, func_X_star, iter_num = damp_newton(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='ILS')
 

