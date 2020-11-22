from goto import with_goto
import utils
import numpy as np
import functions
import functools
import exact_line_search as ELS
import inexact_line_search as ILS
import copy
from GLL import GLL_search
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')

logger = logging.getLogger(__name__)
def descent_by_general_inverse(X, L, D, gfunc):
    """ 方法b：使用广义逆计算D的特征值有负值情况下的下降方向

    Args:
        X ([np.array]): Input X
        L ([np.array]): BP or LDLT分解成的L
        D ([np.array]): BP or LDLT分解成的D
        gfunc ([回调函数]): [目标函数的一阶导函数]
    """

    n = len(D)
    D_plus = np.zeros((n ,n))
    i = 0
    while i < n:
        if i < n - 1 and D[i + 1][i] != 0: #2 * 2的块
            eigenvalue, eigenvector = np.linalg.eig(D[i: i + 2, i: i + 2])
            positive_value_idx = np.where(eigenvalue > 0)[0]
            D_plus[i: i + 2, i: i + 2] = np.dot((eigenvector[positive_value_idx] / eigenvalue[positive_value_idx]).reshape(2,1), eigenvector[positive_value_idx].reshape(1,2))
            i += 2
        else: # 1 * 1的块
            D_plus[i][i] = 0 if D[i][i] <= 0 else 1 / D[i][i]
            i += 1
    L_inverse = np.mat(np.linalg.inv(L))
    descent = -L_inverse.T * np.mat(D_plus) * L_inverse * gfunc(X).reshape(n, 1)
    return np.array(descent)

def negative_condition(x, L, D, diff_function):
    # 特征值存在负数的情况
    logger.info("特征值存在负数的情况")
    n = len(L)
    aT = []    # a是一个行数为n的列向量，要求转置
    i = 0
    while i < n:
        if i == n-1:
            # 到了最后一个1*1主元的情况
            if D[i][i] <= 0:
                aT.append(1)
            else:
                aT.append(0)
            i = i + 1
            break
        if D[i][i+1] == 0 and D[i+1][i] == 0:
            # 1*1主元的情况
            if D[i][i] <= 0:
                aT.append(1)
            else:
                aT.append(0)
            i = i + 1
        else:
            tmp_matrix = [[D[i][i], D[i][i+1]], [D[i+1][i], D[i+1][i+1]]]
            value, array = np.linalg.eig(tmp_matrix)
            value = value.tolist()
            array = array.tolist()
            for index, element in enumerate(value):
                if element < 0:
                    a_tmp = array[index]
                    break
            if len(a_tmp) != 2:
                raise Exception("单位特征向量的维度不对！！！")
            aT.append(a_tmp[0])
            aT.append(a_tmp[1])
            i = i + 2

    a = np.mat(copy.deepcopy(aT))
    a = a.T
    LT = np.mat(copy.deepcopy(L))
    LT = LT.T
    LT_1 = np.linalg.inv(copy.deepcopy(LT))
    t = np.dot(LT_1, a)

    gkt = np.dot(diff_function(x), t)
    if gkt <= 0:
        descent = copy.deepcopy(t)
    else:
        descent = -copy.deepcopy(t)
    descent = np.mat(descent).T    # 转置为行向量
    return descent
    
@with_goto
def Fletcher_Freeman(X, func, gfunc, hess_funct, hyper_parameters=None, search_mode="ELS", epsilon=1e-5, max_epoch=1000):
    """Fletcher_Freeman方法求极小值点

    Args:
        X ([np.array]): [Input X]
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        hess_funct ([回调函数]): [目标函数的Hessian矩阵]
        hyper_parameters: (json): 超参数，超参数中包括：
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
    k = 0
    func_values = [] #记录每一步的函数值，在GLL中有用
    mk = 0 #GLL当中的mk初始值

    label .step2
    G = hess_funct(X)
    func_values.append(func(X))
    L, D, y = utils.Bunch_Parlett(G)
    # logger.info("Dm is ")
    # logger.info(D)
    # from scipy.linalg import ldl
    # L, D, y = ldl(np.array(G, dtype=float), lower=1)
    # logger.info("D is ")
    # logger.info(Dm)

    n = len(X)
    # 根据D的特征值正负性的不同情况，分情况计算下降方向d
    eigenvalue, eigenvector = np.linalg.eig(D)
    
    # 特征值中有负值
    if np.any(eigenvalue < 0):
        logger.info("特征值中有负值")
        d = np.squeeze(descent_by_general_inverse(X, L, D, gfunc))
        
    elif np.any(eigenvalue == 0): # 特征值中既有正值又有零
        logger.info("特征值中既有正值又有零")
        d = descent_by_general_inverse(X, L, D, gfunc)
        if np.where(d != 0)[0].shape[0] == 0:
            G_modified = np.dot(np.dot(L, D), L.T)
            right_zero = np.zeros(n)
            descent_list = np.linalg.solve(G, right_zero) 
            # descent_list = np.linalg.solve(G, right_zero) 
            for descent in descent_list:
                if gfunc(X) @ descent < 0:    # 判断哪一个dk，使得gkdk小于0，把dk为0向量的情况排除出去
                    d = descent
                    break
        
    else:
        logger.info("特征值全为正")
        
        G_modified = np.dot(np.dot(L, D), L.T)
        inv_hass = np.linalg.inv(G)
        # inv_hass = np.linalg.inv(G)
        d = -np.dot(inv_hass , gfunc(X))
    
    #求得下降方向之后，此后的步骤与GM稳定牛顿法无异
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
        logger.info("因为函数值下降在{epsilon}以内，{mode}的FF方法，迭代结束，迭代轮次{iter}，最终X={X}，最终函数值={func_X_new}".format(epsilon=epsilon, mode=search_mode, iter=k,X=X,func_X_new=func_X_new))
        return X_new, func_X_new, k
    if k > max_epoch:
        raise Exception("超过最大迭代次数：{}".format(max_epoch))
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
    
    # logger.info("精确线搜索下的FF方法")
    # Fletcher_Freeman(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='ELS')
    # logger.info("非精确线搜索下的FF方法")
    # Fletcher_Freeman(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='ILS')
    logger.info("GLL线搜索下的FF方法")
    Fletcher_Freeman(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='GLL')
