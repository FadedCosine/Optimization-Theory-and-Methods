import numpy as np
from goto import with_goto
import copy
import functions 
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')

logger = logging.getLogger(__name__)
@with_goto
def retreat_method(func, X, d, hyper_parameters=None, a0=1e-4, r=1e-5, t=1.5):
    """进退法确定初始步长搜索区间 《数值最优化方法》 高立著 p26

    Args:
        func ([函数对象]): [目标函数]
        X ([np.array]]): [初值点]
        d ([np.array]]): [下降方向]
        a0 ([float]]): [初始步长]
        hyper_parameters: (Dic): 超参数，超参数中包括：
            r ([float]): [步长更新的步长]
            t ([float]): [>1的放大率]]
    """
    if hyper_parameters is not None:
        r = hyper_parameters["r"]
        t = hyper_parameters["t"]
    #步1
    assert a0 >=0 and r > 0 and t > 1, "must have a0 >=0 , r > 0 , t > 1"
    i = 0
    alpha = a0
    a_pre = a0
    just_change_direction_flag = False
    func_k = 1
    func_pre = func(X + d * a_pre)
    #步2
    label .step2
    
    a_cur = a_pre + r
    
    if a_cur <= 0:
        a_cur = 0
        goto .step4

    func_k += 1
    func_cur = func(X + d * a_cur)
    if func_pre <= func_cur:
        # 可能会存在两个方向均是不是下降方向的情况
        if just_change_direction_flag:
            logger.info("陷入鞍点")
            return a_pre, a_pre, func_k
        #转步4
        goto .step4
        
    #步3
    r = t * r
    alpha = a_pre
    a_pre = a_cur
    func_pre = func_cur
    i += 1
    goto .step2
    
    label .step4
    if i == 0:
        r = -r
        alpha = a_cur
        just_change_direction_flag = True
        #转步2
        goto .step2
    else:
        return min(alpha, a_cur), max(alpha, a_cur), func_k


@with_goto
def golden_method(func, X, d, a0, b0, hyper_parameters=None, epsilon=1e-5, tau=0.618):
    """0.618法确定函数近似极小点 《最优化理论与方法》 袁亚湘著 p71

    Args:
        func ([函数对象]): [目标函数]
        X ([np.array]]): [初值点]
        d ([np.array]]): [下降方向]
        a0 ([float]]): [步长区间下界]
        b0 ([float]]): [步长区间上界]
        hyper_parameters: (Dic): 超参数，超参数中包括：
            epsilon ([float]): [终止条件阈值]
        tau ([float]): [0.618]]
    """
    if hyper_parameters is not None:
        epsilon = hyper_parameters["epsilon"]
    if a0 == b0:
        return a0, 0
    assert b0 > a0 and epsilon > 0, "must have b0 > a0, epsilon > 0"
    a, b = a0, b0
    #步1
    al = a + (1 - tau) * (b -a)
    ar = a + tau * (b - a)
    func_k = 2
    f_al = func(X + d * al)
    f_ar = func(X + d * ar)
    
    #步2
    label .step2
    if f_al <= f_ar:
        goto .step4
    #步3
    if b - al <= epsilon:
        return ar, func_k
    else:
        a = al
        al = ar
        f_al = f_ar
        ar = a + tau * (b - a)
        func_k += 1
        f_ar = func(X + d * ar)
        goto .step2
    #步4
    label .step4
    if ar - a <= epsilon:
        return al, func_k
    else:
        b = ar
        ar = al
        f_ar = f_al
        al = a + (1 - tau) * (b - a)
        func_k += 1
        f_al = func(X + d * al)
        goto .step2



