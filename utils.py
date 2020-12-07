import math 
import copy
import numpy as np
from goto import with_goto
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

@with_goto
def modified_Cholesky(G, hyper_parameters=None, u=1e-20):
    """修正Cholesky分解

    Args:
        G ([np.array]): 用于分解的二维矩阵
        hyper_parameters: (Dic): 超参数，超参数中包括：
            u:  机器精度
    """
    if hyper_parameters is not None:
        u = hyper_parameters['u']
    # 步1：初始化
    G = np.array(G)
    gamma = 0 # 对角元最大元素
    ksai = 0 # 非对角元最大元素
    n = len(G)
    for i in range(n):
        for j in range(n):
            if i == j:
                gamma = max(gamma, abs(G[i][i])) 
            else:
                ksai = max(ksai, abs(G[i][j]))
    beta_2 = max(gamma, ksai / math.sqrt(n ** 2 - 1), u)
    delta = u * max(gamma + ksai, 1)
    
    assert delta > 0 , "must have delta > 0" 
    L = np.eye(n, dtype=float)
    D = np.zeros((n,n), dtype=float)
    C = np.zeros((n,n), dtype=float)
    #按列计算
    j = 1 #表示当前计算的列的indx
    # 步2：计算dj'
    label .step2
    dj_prime = max(delta, abs(G[j - 1][j - 1] - sum((C[j - 1][r - 1] ** 2 / (D[r - 1][r - 1]) for r in range(1, j))) ) )  
    # 步3：计算Cij
    for i in range(j + 1, n + 1):
        C[i - 1][j - 1] = G[i - 1][j - 1] - sum(( L[j - 1][r - 1] * C[i - 1][r - 1] for r in range(1, j)))
    
    # 步4：计算theta_j
    theta_j = 0
    if j < n:
        theta_j = max(( abs(C[i - 1][j - 1]) for i in range(j + 1, n + 1)))
    # 步5：计算d_j
    D[j - 1][j - 1] = max(dj_prime, theta_j ** 2 / beta_2)
    # 步6：计算l_ij
    for i in range(j + 1, n + 1):
        L[i - 1][j - 1] = C[i - 1][j - 1] / D[j - 1][j - 1]
    # 步7，更新j，判断是否终止
    if j + 1 <= n:
        j += 1
        goto.step2
    else:
        return L, D

def get_modified_G(L, D):
    LT = copy.deepcopy(L).T
    C = np.dot(L, D)
    return np.dot(C, LT)

@with_goto
def Bunch_Parlett(A):
    """对A进行BP分解，输出DL

    Args:
        A ([np.array]): 输入的矩阵
    """
    # 步1：初始化
    A_ = copy.deepcopy(A)
    n = len(A)
    D = np.zeros((n, n))
    L = np.zeros((n, n))
    #记录变量顺序
    y = np.array(range(n))
    k, m = 1, 0
    # 步2：求n-m阵中对角元中的最大值
    label .step2
    a_tt = 0
    t = -1
    for i in range(m, n):
        if abs(A_[i][i]) > a_tt:
            a_tt = abs(A_[i][i])
            t = i
    # 步3：求n-m阵中非对角元的最大值
    a_ls = 0
    l, s = -1, -1
    if m < n - 1:
        for i in range(m, n):
            for j in range(m, i):
                if abs(A_[i][j]) > a_ls:
                    a_ls = abs(A_[i][j])
                    l = i
                    s = j
    
    # 步4：根据对角元最大值和非对角元最大值比较，判断分支
    if a_tt == 0 and a_ls == 0:
        goto .step8
    elif a_tt < 2.0 / 3 * a_ls:
        goto .step6
    # 步5：1*1 的块
    # print("第{k}步是 1 * 1的块:".format(k=k))
    # print("第{k}步最初的A:".format(k=k))
    # print(A_)
    # 交换行列
    A_[[m, t], :] = A_[[t, m], :]
    A_[:, [m, t]] = A_[:, [t, m]]
    # print("交换行列后的A:".format(k=k))
    # print(A_)
    # y也要交换行列
    y[m], y[t] = y[t], y[m]
    # L也要交换行列
    L[[m, t], :] = L[[t, m], :]
    L[:, [m, t]] = L[:, [t, m]]
    # 对D对应位置的元素赋值
    D[m][m] = A_[m][m]
    L[m][m] = 1
    
    L[m + 1:, m] = A_[m + 1:, m] / A_[m][m] # 进行了操作之后就赋值了新的空间了，不用再deepcopy
    # print(np.dot((L[m:, m] * D[m][m]).reshape(n-m,1) , L[m:, m].reshape(1,n-m)))
    A_[m:, m:] -= np.dot((L[m:, m] * D[m][m]).reshape(n-m,1) , L[m:, m].reshape(1,n-m))
    m += 1
    # print("消解之后A是")
    # print(A_)
    # print("消解之后L是")
    # print(L)
    # print("消解之后D是")
    # print(D)
    goto .step7
    # 步6：2*2 的块
    label .step6
    # 因为l > s，所有l行放在m+1行，l列放在m+1列，s行放在m行，s列放在m列
    # 注意当m+1 == s的时候，直接用一行内写交换行列的代码可能会存在问题,所以一定要先写交换 行m和行s的代码
    A_[[m, s], :] = A_[[s, m], :]
    A_[[m + 1, l], :] = A_[[l, m+1], :]
  
    A_[:, [m, s]] = A_[:, [s, m]]
    A_[:, [m + 1, l]] = A_[:, [l, m + 1]]
    # L也要交换行列
    L[[m, s], :] = L[[s, m], :]
    L[[m + 1, l], :] = L[[l, m+1], :]
  
    L[:, [m, s]] = L[:, [s, m]]
    L[:, [m + 1, l]] = L[:, [l, m + 1]]
    # print("交换行列后的A:".format(k=k))
    # print(A_)
    y[m], y[s] = y[s], y[m]
    y[m + 1], y[l] = y[l], y[m + 1], 
    # 对D对应位置的元素赋值
    D[m: m + 2, m: m + 2] = copy.deepcopy(A_[m: m + 2, m: m + 2])
    L[m: m + 2, m: m + 2] = np.eye(2) # 二阶单位阵
    # print("交换行列后的A是")
    # print(A_)
    L[m + 2:, m: m + 2] = np.dot(A_[m + 2:, m: m + 2] , np.linalg.inv(A_[m: m + 2, m: m + 2]))
    
    A_[m:, m:] -= np.dot( np.dot(L[m:, m: m + 2].reshape(n-m,2) , D[m: m + 2, m: m + 2]), np.mat(L[m:, m: m + 2]).T)
    m += 2
    
    # print("消解之后A是")
    # print(A_)
    # print("消解之后L是")
    # print(L)
    # print("消解之后D是")
    # print(D)
    # 步7：
    label .step7
    if m < n:
        k += 1
        goto .step2
    # 步8
    label .step8
    return L, D, y


if __name__ == '__main__':
    G = np.array([[1, 1,       2], 
                  [1, 1+1e-20, 3],
                  [2, 3,       1]])
    # print("修正Cholesky分解")
    # L, D = modified_Cholesky(G)
    # G_ = get_modified_G(L, D)
    # print("L 是：")
    # print(L)
    # print("D 是：")
    # print(D)
    # print("修正过的G 是：")
    # print(G_)
    # G_1 = np.linalg.inv(G_)
    # print(G_1)

    # print("BP分解")
    # L, D, y= Bunch_Parlett(G)
    # G_ = get_modified_G(L, D)
    # print("L 是：")
    # print(L)
    # print("D 是：")
    # print(D)
    # print("修正过的G 是：")
    # print(G_)

    # G = np.array([[11202, 1200, 0, 0], 
    #                 [1200, 220.200000000000, 0, 19.8000000000000], 
    #                 [0, 0, 10082, 1080], 
    #                 [0, 19.8000000000000, 1080, 200.200000000000]])
    G = np.array(
        [[1, 1, 2],
         [1, 2, 3],
         [2, 3, 1]],
    dtype = float
    )
    print("BP分解")
    L, D, y= Bunch_Parlett(G)
    G_ = get_modified_G(L, D)
    print("L 是：")
    print(L)
    print("D 是：")
    print(D)
    print("修正过的G 是：")
    print(G_)
    # from scipy.linalg import ldl
    # lu, d, perm = ldl(np.array(G, dtype=float), lower=1)
    # print("LDL 的 L是 ")
    # print(lu[perm, :])
    # print("LDL 的 D是")
    # print(d)
    # G_ = get_modified_G(lu, d)
    # print("修正过的G 是：")
    # print(G_)