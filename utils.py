import math 
import copy
import numpy as np
from goto import with_goto

@with_goto
def modified_Cholesky(G, u=1e-10):
    """修正Cholesky分解

    Args:
        G ([np.array]): 用于分解的二维矩阵
    """
    # 步1：初始化
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


if __name__ == '__main__':
    L, D = modified_Cholesky([[1, 1, 2], [1, 1+1e-20, 3], [2, 3, 1]])
    G_ = get_modified_G(L, D)
    print(L)
    print(D)
    print(G_)
    G_1 = np.linalg.inv(G_)
    print(G_1)

    # L, D = modified_Cholesky([[11202, 1200, 0, 0], [1200, 240, 0, 0], [0, 0, 10082, 1080], [0, 0, 1080, 220]])
    # G_ = get_modified_G(L, D)
    # print(L)
    # print(D)
    # print(G_)