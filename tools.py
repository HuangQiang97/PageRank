import numpy as np
from scipy.sparse import rand


def generate_sparse_matrix(n, density=0.1):
    """
    生成一个稀疏随机方阵，元素值为0或1。

    参数:
    - n: 矩阵的行数/列数
    - density: 矩阵的稀疏度，范围从0到1，默认为0.1。

    返回:
    - 一个稀疏方阵，其元素值为0或1。
    """
    S = rand(n, n, density, format='csr', random_state=47)
    M = np.array(S.A.round())
    column_sum = np.sum(M, axis=0)
    M = M / np.sum(M, axis=0)
    M[:, column_sum == 0] = 0.0

    with open('./data/input.txt', 'w') as f:
        for j in range(n):
            adj_nodes = []
            for i in range(n):
                if M[i, j] != 0:
                    adj_nodes.append(i + 1)
            f.write('%d\t[[%s], %f]\n' % (j + 1, ','.join(map(str, adj_nodes)), 1 / n))

    return M


def check_err(M, pagerank_vector, d=0.85):
    """
    计算平稳分布向量稳态误差

    参数:
    - M: 状态转移矩阵
    - pagerank_vector: 平稳分布向量
    - d: 阻尼因子

    返回:
    - 稳态误差
    """
    n = M.shape[0]
    new_vector = (1 - d) / n + d * M @ pagerank_vector
    err = np.linalg.norm(new_vector - pagerank_vector, 1)
    return err


def load_data():
    file_path = './data/input.txt'
    with open(file_path, 'r') as f:
        n = len(f.readlines())

    M = np.zeros((n, n))
    # 读取文件并更新矩阵
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split('\t')
            col = int(parts[0])  # 列索引
            row_indices = eval(parts[1])[0]
            for row in row_indices:
                M[row - 1, col - 1] = 1 / len(row_indices)
    return M


def matrix_multiply_and_sum(A_blocks, X_values, B_value):
    # 假设 A_blocks 是 (行索引, 列索引, 值) 组成的列表
    # X_values 是 (索引, 值) 组成的列表
    # B_value 是单个浮点数（实际使用中可能是向量）

    # 从 A_blocks 和 X_values 构建 numpy 数组
    n = int(np.sqrt(len(A_blocks)))  # 假设 A 是 n x n 矩阵
    A = np.zeros((n, n))
    for i, j, val in A_blocks:
        A[i, j] = val

    X = np.zeros(n)
    for i, val in X_values:
        X[i] = val

    # 执行矩阵乘法
    AX = np.dot(A, X)

    # 将结果与 B 相加
    Y = AX + B_value

    return Y

if __name__ == '__main__':
    n = 8
    density = 0.4  # 矩阵的稀疏度
    M = generate_sparse_matrix(n, density)  # 转移矩阵
    print(M)
