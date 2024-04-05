import time

import numpy as np
import scipy

from tools import check_err, load_data


def pagerank_iter(M, d, tol=1.0e-6):
    """
    幂法求解马尔可夫平稳分布向量

    参数:
    - M: 状态转移矩阵
    - d: 阻尼因子
    - tol: 迭代停止阈值

    返回:
    - 平稳分布向量
    """

    n = M.shape[0]
    pagerank_vector = np.ones(n) / n
    base = (1 - d) / n * np.ones(n)

    while (True):
        new_rank = d * M @ pagerank_vector + base
        if np.linalg.norm(new_rank - pagerank_vector, ord=1) < tol:
            return new_rank
        pagerank_vector = new_rank


def pagerank_algebraic(M, d):
    """
    代数法求解马尔可夫平稳分布向量

    参数:
    - M: 状态转移矩阵
    - d: 阻尼因子

    返回:
    - 平稳分布向量
    """

    I = np.eye(n)
    coeff_matrix = I - d * M
    b = np.ones(n) * (1 - d) / n
    pagerank_vector = scipy.linalg.solve(coeff_matrix, b)

    return pagerank_vector


if __name__ == '__main__':
    M = load_data()
    n = M.shape[0]
    d = 0.85  # 阻尼系数

    # 迭代法
    start = time.time_ns()
    pagerank_vector = pagerank_iter(M, d)
    print('time cost: %f ms' % ((time.time_ns() - start) / 1e6))
    print('pagerank:%s' %['%.5f'%x for x in  pagerank_vector])
    print('err: %E' % check_err(M, pagerank_vector, d))
    # 代数法
    start = time.time_ns()
    pagerank_vector = pagerank_algebraic(M, d)
    print('time cost: %f ms' % ((time.time_ns() - start) / 1e6))
    print('pagerank:%s' %['%.5f'%x for x in  pagerank_vector])
    print('err: %E' % check_err(M, pagerank_vector, d))
