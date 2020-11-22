import functions
import numpy as np
from goto import with_goto
import exact_line_search as ELS
import inexact_line_search as ILS
import utils
import functools
import copy
from GLL import GLL_search
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')

logger = logging.getLogger(__name__)
@with_goto
def armijo_goldstein(x0, function, diff_function, descent, a=1e-6, p=0.1, t=5):
    # 输入：x0为当前迭代点
    # function用于求函数值
    # diff_function用于求导数值
    # descent是当前步的下降方向
    # 输出：满足Armijo-Goldstein准则的非精确线搜索的步长
    a1 = 0
    a2 = 10 ** 10
    a_k = copy.deepcopy(a)
    value_diff = diff_function(x0)
    value = function(x0)

    label.step2_ag
    # 第二步
    x = copy.deepcopy(x0)
    for element in range(len(x)):
        x[element] += a_k * descent[element]
    gkt_dk = 0    # 计算gkT*dk的值
    for element in range(len(value_diff)):
        gkt_dk += value_diff[element] * descent[element]
    if function(x) <= value + p * a_k * gkt_dk:    # 判断是否满足Armijo准则？
        pass
    else:
        goto.step4_ag

    # 第三步
    x = copy.deepcopy(x0)
    for element in range(len(x)):
        x[element] += a_k * descent[element]
    gkt_dk = 0    # 计算gkT*dk的值
    for element in range(len(value_diff)):
        gkt_dk += value_diff[element] * descent[element]
    if function(x) >= value + (1 - p) * a_k * gkt_dk:    # 判断是否满足Goldstein准则？
        return a_k
    else:
        a1 = copy.deepcopy(a_k)
        if a2 < 10 ** 10:
            a_k = (a1 + a2) / 2
        else:
            a_k = t * a_k
        goto.step2_ag

    label.step4_ag
    # 第四步
    a2 = copy.deepcopy(a_k)
    a_k = (a1 + a2) / 2
    goto.step2_ag


@with_goto
def damp_newton(X, func, gfunc, hess_funct, hyper_parameters=None, search_mode="ELS", use_modified_Cholesky=True, epsilon=1e-5, max_epoch=1000):
    """[使用阻尼牛顿法极小值点
         d = -G_k^{-1} * g_k]

    Args:
        X ([np.array]): [Input X]
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        hess_funct ([回调函数]): [目标函数的Hessian矩阵]
        hyper_parameters: (Dic): 超参数，超参数中包括：
            search_mode (str, optional): [线搜索的模式（选择精确线搜索还是非精确线搜索）]. Defaults to 'ELS'. ['ELS', 'ILS']
            epsilon ([float], optional): [当函数值下降小于epsilon，迭代结束]. Defaults to 1e-5.
            max_epoch (int, optional): [最大允许的迭代次数]. Defaults to 1000.

    Returns:
        返回求解得到的极小值点，极小值点对应的函数值和迭代次数
    """
    if hyper_parameters is not None:
        search_mode = hyper_parameters["search_mode"]
        epsilon = hyper_parameters["epsilon"]
        max_epoch = hyper_parameters["max_epoch"]
        use_modified_Cholesky = hyper_parameters["damp_newton"]["use_modified_Cholesky"]

    k = 1
    func_values = [] #记录每一步的函数值，在GLL中有用
    mk = 0 #GLL当中的mk初始值
    #计算下降方向d_k
    label .count_dk
    G = hess_funct(X)
    g = gfunc(X)
    # 把当前函数值加入func_values
    func_values.append(func(X))
    try:
        if use_modified_Cholesky:
            L, D = utils.modified_Cholesky(G, hyper_parameters["modified_Cholesky"])
            G_ = utils.get_modified_G(L, D)
            inv_hass = np.linalg.inv(G_)
            d = -np.dot(inv_hass , g)
        else:
            inv_hass = np.linalg.inv(G)
            d = -np.dot(inv_hass , g)
    except:
        logger.info("Hessian 矩阵不可逆，用修正Cholesky分解求下降方向")
        L, D = utils.modified_Cholesky(G, hyper_parameters["modified_Cholesky"])
        G_ = utils.get_modified_G(L, D)
        inv_hass = np.linalg.inv(G_)
        d = -np.dot(inv_hass , g)
    
    #计算步长
    if search_mode == "ELS":
        logger.info("迭代第{iter}轮，当前X取值为{X}，下降方向为{d}，当前函数值为{func_x}".format(iter=k,X=X,d=d,func_x=round(func(X), 5)))
        [a, b] = ELS.retreat_method(func, X, d, hyper_parameters=hyper_parameters["ELS"]["retreat_method"] if hyper_parameters is not None else None) 
        alpha_star = ELS.golden_method(func, X, d, a, b, hyper_parameters=hyper_parameters["ELS"]["golden_method"] if hyper_parameters is not None else None) 
    elif search_mode == "ILS":
        logger.info("迭代第{iter}轮，当前X取值为{X}，下降方向为{d}，当前函数值为{func_x}".format(iter=k,X=X,d=d,func_x=round(func(X), 5)))
        # alpha_star = armijo_goldstein(X, func, gfunc, d)
        alpha_star = ILS.inexact_line_search(func, gfunc, X, d, hyper_parameters=hyper_parameters["ILS"] if hyper_parameters is not None else None) 
    elif search_mode == "GLL":
        logger.info("迭代第{iter}轮，当前X取值为{X}，下降方向为{d}，当前函数值为{func_x}".format(iter=k,X=X,d=d,func_x=round(func(X), 5)))
        alpha_star, mk = GLL_search(func, gfunc, X, d, func_values, mk, hyper_parameters=hyper_parameters["GLL"] if hyper_parameters is not None else None) 
    else:
        raise ValueError("参数search_mode 必须从['ELS', 'ILS']当中选择")
    
    X_new = X + d * alpha_star
    func_X_new = func(X_new)
    if abs(func_X_new - func(X)) <= epsilon:
        logger.info("因为函数值下降在{epsilon}以内，{mode}的阻尼牛顿法，迭代结束，迭代轮次{iter}，最终X={X}，最终函数值={func_X_new}".format(epsilon=epsilon, mode=search_mode,iter=k, X=X,func_X_new=func_X_new))
        return X_new, func_X_new, k
    if k > max_epoch:
        raise Exception("超过最大迭代次数：%d", max_epoch)
    X = X_new
    k += 1
    goto .count_dk



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
def GM_newton(X, func, gfunc, hess_funct, hyper_parameters=None, zeta=1e-2, search_mode="ELS", epsilon=1e-5, max_epoch=1000):
    """使用Gill Murray稳定牛顿法求极小值点
         d = -G_k^{-1} * g_k]

    Args:
        X ([np.array]): [Input X]
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        hess_funct ([回调函数]): [目标函数的Hessian矩阵]
        hyper_parameters: (Dic): 超参数，超参数中包括：
            zeta ([float], optional): [当gk的模大于zeta， 求解方程得到下降方向]. Defaults to 1e-2.
            search_mode (str, optional): [线搜索的模式（选择精确线搜索还是非精确线搜索）]. Defaults to 'ELS'. ['ELS', 'ILS']
            epsilon ([float], optional): [当函数值下降小于epsilon，迭代结束]. Defaults to 1e-5.
            max_epoch (int, optional): [最大允许的迭代次数]. Defaults to 1000.

    Returns:
        返回求解得到的极小值点，极小值点对应的函数值和迭代次数
    """

    if hyper_parameters is not None:
        zeta = hyper_parameters["GM_newton"]["zeta"]
        search_mode = hyper_parameters["search_mode"]
        epsilon = hyper_parameters["epsilon"]
        max_epoch = hyper_parameters["max_epoch"]

    k = 1
    func_values = [] #记录每一步的函数值，在GLL中有用
    mk = 0 #GLL当中的mk初始值
    assert epsilon > 0 , "must have epsilon > 0" 
    # 步2：计算g和G
    label .step2
    g = gfunc(X)
    G = hess_funct(X)
    # 把当前函数值加入func_values
    func_values.append(func(X))
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
        logger.info("因为负曲率方向不存在，{mode}的GM稳定牛顿法，迭代结束，迭代轮次{iter}，最终X={X}，最终函数值={func_X_new}".format(mode=search_mode,iter=k,X=X,func_X_new=func_X_new))
        return X, func(X), k
    else:
        gT = np.mat(g).T
        if np.dot(gT, d) > 0:
            d = -d
    # 步6：线搜索求步长
    label .step6
    if search_mode == "ELS":
        logger.info("迭代第{iter}轮，当前X取值为{X}，下降方向为{d}，当前函数值为{func_x}".format(iter=k,X=X,d=d,func_x=round(func(X), 5)))
        [a, b] = ELS.retreat_method(func, X, d, hyper_parameters=hyper_parameters["ELS"]["retreat_method"] if hyper_parameters is not None else None) 
        alpha_star = ELS.golden_method(func, X, d, a, b, hyper_parameters=hyper_parameters["ELS"]["golden_method"] if hyper_parameters is not None else None) 
    elif search_mode == "ILS":
        logger.info("迭代第{iter}轮，当前X取值为{X}，下降方向为{d}，当前函数值为{func_x}".format(iter=k,X=X,d=d,func_x=round(func(X), 5)))
        alpha_star = ILS.inexact_line_search(func, gfunc, X, d, hyper_parameters=hyper_parameters["ILS"] if hyper_parameters is not None else None) 
    elif search_mode == "GLL":
        logger.info("迭代第{iter}轮，当前X取值为{X}，下降方向为{d}，当前函数值为{func_x}".format(iter=k,X=X,d=d,func_x=round(func(X), 5)))
        alpha_star, mk = GLL_search(func, gfunc, X, d, func_values, mk, hyper_parameters=hyper_parameters["GLL"] if hyper_parameters is not None else None) 
    else:
        raise ValueError("参数search_mode 必须从['ELS', 'ILS']当中选择")

    X_new = X + d * alpha_star
    func_X_new = func(X_new)
    if abs(func_X_new - func(X)) <= epsilon:
        logger.info("因为函数值下降在{epsilon}以内，{mode}的GM稳定牛顿法，迭代结束，迭代轮次{iter}，最终X={X}，最终函数值={func_X_new}".format(mode=search_mode,epsilon=epsilon, iter=k,X=X,func_X_new=func_X_new))
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
    # logger.info("精确线搜索下的阻尼牛顿法")
    # X_star, func_X_star, iter_num = damp_newton(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='ELS')
    # logger.info("非精确线搜索下的阻尼牛顿法")
    # X_star, func_X_star, iter_num = damp_newton(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='ILS')
    logger.info("GLL线搜索下的阻尼牛顿法")
    X_star, func_X_star, iter_num = damp_newton(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='GLL')
    
    # logger.info("精确线搜索下的GM稳定牛顿法")
    # X_star, func_X_star, iter_num = GM_newton(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='ELS')
    # logger.info("非精确线搜索下的GM稳定牛顿法")
    # X_star, func_X_star, iter_num = GM_newton(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='ILS')
    # logger.info("GLL线搜索下的GM稳定牛顿法")
    # X_star, func_X_star, iter_num = GM_newton(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='GLL')


