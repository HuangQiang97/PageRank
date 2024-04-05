import random

import numpy as np

np.random.seed(114514)

# 分块计算Y=M@PR+B
n = 128
ki, kj = 7, 11  # ki,kj为分割的块数
M = np.random.randn(n * n).reshape(n, n)
PR = np.random.randn(n, 1).reshape(n, 1)
B = np.ones((n, 1)) * (1 - 0.85) / n

# 生成横向和纵向的分割点
block_i = sorted([0] + random.sample(range(1, n), ki - 1) + [n])
block_j = sorted([0] + random.sample(range(1, n), kj - 1) + [n])

Y = np.zeros_like(B)  # 初始化Y

# 分块处理
for i in range(ki):
    start_i = block_i[i]
    end_i = block_i[i + 1]
    for j in range(kj):
        start_j = block_j[j]
        end_j = block_j[j + 1]
        Y[start_i:end_i] += M[start_i:end_i, start_j:end_j] @ PR[start_j:end_j]
Y += B  # 不要忘记加上B

print(np.linalg.norm(Y - (M @ PR + B)))
