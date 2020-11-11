import functions
import numpy as np
import numpy.linalg.LinAlgError as LinAlgError
def basic_newton_for_d(gfunc, hess_funct, X):
    """[使用基本牛顿法求得下降方向 d = -G_k^{-1} * g_k]

    Args:
        gfunc ([type]): [目标函数的一阶导函数]
        hess_funct ([type]): [目标函数的Hessian矩阵]
        X ([np.array]): [Input X]
    """
    try:
        inv_hass = np.linalg.inv(hess_funct(X))
        return -inv_hass * gfunc(X)
    except LinAlgError:
        print("Hessian 矩阵不可逆，尝试其他方法求下降方向")
        # TODO: 加上修正Cholesky计算下降方向的参数
        return modified_Cholesky_for_d()

# TODO：实现修正Cholesky计算下降方向
def modified_Cholesky_for_d():
    return
    

