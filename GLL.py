from goto import with_goto

def GLL_search(func, gfunc, X, d, hyper_parameters=None, M=10, a=0.5, sigma=0.5, rho=0.1):
    """ 非单调线搜索GLL准则

    Args:
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        X ([np.array]]): [初值点]
        d ([np.array]]): [下降方向]
        hyper_parameters: (Dic): 超参数，超参数中包括：
            M (int, optional): [用于限制m(k)上限的参数]. Defaults to 10.
            a (int, optional): [初始步长]. Defaults to 0.5.
            sigma (int, optional): [用于确定初始步长的在0到1之间的系数]. Defaults to 0.5.
            rho (float, optional): [GLL准则当中的参数]. Defaults to 0.1.
    Returns:
        [float]: [搜索得到的步长]]
    """
    if hyper_parameters is not None:
        rho = hyper_parameters["rho"]
        sigma = hyper_parameters["sigma"]
        M = hyper_parameters["M"]
        a = hyper_parameters["a"]
        # alpha = hyper_parameters["alpha0"]
    f0, gf0 = func(X), gfunc(X)
    gkdk = gf0.dot(d)