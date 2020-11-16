import numpy as np
from goto import with_goto
import copy
import functions 

@with_goto
def retreat_method(func, X, d, a0=0, r=1e-5, t=1.5):
    """进退法确定初始步长搜索区间 《数值最优化方法》 高立著 p26

    Args:
        func ([函数对象]): [目标函数]
        X ([np.array]]): [初值点]
        d ([np.array]]): [下降方向]
        a0 ([float]]): [初始步长]
        r ([float]): [步长更新的步长]
        t ([float]): [>1的放大率]]
    """
    #步1
    assert a0 >=0 and r > 0 and t > 1, "must have a0 >=0 , r > 0 , t > 1"
    i = 0
    alpha = a0
    a_pre = a0
    just_change_direction_flag = False
    #步2
    label .step2
 
    a_cur = a_pre + r
    
    if a_cur <= 0:
        a_cur = 0
        goto .step4
    
    if func(X + d * a_pre) <= func(X + d * a_cur):
        # 可能会存在两个方向均是下降方向的情况
        if just_change_direction_flag:
            return [0, 0]
        #转步4
        goto .step4
        
    
    #步3
    r = t * r
    alpha = a_pre
    a_pre = a_cur
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
        return [min(alpha, a_cur), max(alpha, a_cur)]


@with_goto
def golden_method(func, X, d, a0, b0, epsilon=1e-5, tau=0.618):
    """0.618法确定函数近似极小点 《最优化理论与方法》 袁亚湘著 p71

    Args:
        func ([函数对象]): [目标函数]
        X ([np.array]]): [初值点]
        d ([np.array]]): [下降方向]
        a0 ([float]]): [步长区间下界]
        b0 ([float]]): [步长区间上界]
        epsilon ([float]): [终止条件阈值]
        tau ([float]): [0.618]]
    """
    if a0 == 0 and b0 == 0:
        return a0
    assert a0 > 0 and b0 > 0 and epsilon > 0, "must have a0 > 0, b0 > 0, epsilon > 0"
    a, b = a0, b0
    #步1
    al = a + (1 - tau) * (b -a)
    ar = a + tau * (b - a)
    f_al = func(X + d * al)
    f_ar = func(X + d * ar)
    #步2
    label .step2
    if f_al <= f_ar:
        goto .step4
    #步3
    if b - al <= epsilon:
        return ar
    else:
        a = al
        al = ar
        f_al = f_ar
        ar = a + tau * (b - a)
        f_ar = func(X + d * ar)
        goto .step2
    #步4
    label .step4
    if ar - a <= epsilon:
        return al
    else:
        b = ar
        ar = al
        f_ar = f_al
        al = a + (1 - tau) * (b - a)
        f_al = func(X + d * al)
        goto .step2

def test():
    x0 = np.array([-3, -1, -3, -1])
    d0 = np.array([2, 1, 2, 1])
    [a, b] = retreat_method(functions.wood, x0, d0, 0, 0.1, 1.5)
    print(a)
    print(b)
    best_x = golden_method(functions.wood, x0, d0, a, b, epsilon=1e-5)
    print("origin x0 is ", x0)
    print(best_x)
    print("best_x is ", x0 + d0 * best_x)


