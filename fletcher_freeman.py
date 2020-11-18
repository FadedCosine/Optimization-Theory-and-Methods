from goto import with_goto
import utils
import numpy as np
import functions
import functools
@with_goto
def Fletcher_Freeman(X, func, gfunc, hess_funct, search_mode='ELS', ILS_model="Wolfe Powell", epsilon=1e-5, max_epoch=1000):
    """Fletcher_Freeman方法求极小值点

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

    G = hess_funct(X)
    L, D, y = utils.Bunch_Parlett(G)

    # 根据D的特征值正负性的不同情况，分情况计算下降方向d
    print(D)
    eigenvalue, eigenvector = np.linalg.eig(D)
    print(eigenvalue)
    print(eigenvector)


if __name__ == '__main__':
    x0 = np.array([-3, -1, -3, -1])
    d0 = np.array([2, 1, 2, 1])
    diff_wood_list, symbols_wood_list = functions.diff_wood_expression()
    g_wood_partial = functools.partial(functions.g_wood, diff_list=diff_wood_list, symbols_list=symbols_wood_list)
    hess_wood_lists, symbols_wood_list = functions.hess_wood_expression()
    G_wood_partial = functools.partial(functions.G_wood, G_lists=hess_wood_lists, symbols_list=symbols_wood_list)
    Fletcher_Freeman(x0, functions.wood, g_wood_partial, G_wood_partial, search_mode='ELS')
