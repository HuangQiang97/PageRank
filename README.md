### PageRank网页排序算法

### 1，定义

将整个互联网视为一个巨大的有向图$G = (V , E) $，网页构成有向图的节点，网页间的超链接构成有向图的边，据此构建一个随机游走模型，即一阶马尔可夫链[^1]。

在这个模型中，为每个网页设定一个初始化的`PageRank`值，表示用户停留在该网页的概率，==网页浏览者会随机地、按照等概率地跟随一个页面上的任何一个超链接到另一个页面，并持续这种随机跳转==。

在长时间内，这种随机跳转的行为会形成一个稳定的模式（马尔可夫链的平稳分布），每个网页的 `PageRank` 值，即用户停留在每个网页的概率收敛到一个稳定值。

直观上，如果指向网页$i$的超链接越多、能跳转到网页$i$的上游网页$j$的`PageRank`越高，随机跳转到网页$i$的概率也就越高，网页$i$的`PageRank`值就越高，网页也就越重要[^2]。 

### 2，数学表示

有向图的状态转移矩阵定义为$\mathbf M=[m_{i,j}]_{n\times n}$，其中$m_{i,j}$表示$\mathbf M$的第$i$行第$j$列，也即节点$j$转移到节点$i$的概率，$\mathbf M$具有如下性质：

* 如果$j,i$间存在边，从节点$j$出发到达节点$i$的概率等于节点$j$出度的倒数，否则转移概率为0。定义$\mathbb v(j)$为节点$j$指向页面的集合，$L(\mathbb v(j))$为节点$j$的出度
  $$
  \begin{equation}
  m_{i,j}=
  \begin{cases}
  \frac{1}{L(\mathbb v(j))} & \text{if } j,i间存在指向关系, \\
  0 & \text{if } j,i间不存在指向关系.
  \end{cases}
  \end{equation}
  $$

* 如果节点$j$出度不为0，从节点$j$出发到达其他节点的概率和为1：$\sum_im_{i,j}=1$
<center><img src="./assets/image-20240403154343674.png" alt="image-20240403154343674" style="zoom:33%;" /></center>


图示[^3]节点转移关系的状态转移矩阵可以表示为
$$
\mathbf M=\begin{equation*}
\begin{bmatrix}
0 & \frac{1}{2} & \frac{1}{3} & 1 \\
0 & 0 & \frac{1}{3} & 0 \\
0 & 0 & 0 & 0 \\
0 & \frac{1}{2} & \frac{1}{3} & 0 \\
\end{bmatrix}
\end{equation*}
$$


定义$t$时刻所有节点的`PageRank`值为向量$\mathbf {R}_t=[r_t^i]_n$，其中$r_t^i$表示$t$时刻停留在节点$i$的概率，满足$\sum_ir_t^i=1$，0时刻的状态分布向量初始化为$\mathbf M_0=[\frac{1}{n}]_{n}$，则$t+k$时刻的状态分布向量
$$
\mathbf R_{t+k}=\mathbf M^k\mathbf R_t
$$
当经过足够长时间，上述随机游走过程达到马尔可夫链的平稳分布，$\mathbf R_t$收敛至$\mathbf R$
$$
\mathbf R=\lim_{t \to \infty} \mathbf M^t\mathbf R_0
$$
此时由于状态已经收敛，满足
$$
\mathbf R=\mathbf {MR}
$$
要保证上述马尔可夫过程具有稳态分布，需要满足以下条件

* 不可约性: 有向图是连通图，没有孤立的节点。

* 非周期性: 对于任何状态，返回到该状态的步数不是一个固定的周期。

* 正常返还性: 每个状态都会被反复访问，且访问的平均间隔时间是有限的。

由节点间链接指向关系构成的随机游走过程不一定同时满足上述三个条件，即不保证存在平稳状态。对此引入常数矩阵$\mathbf E/n$，其中$\mathbf E$为$n\times n$全1矩阵，状态转移过程更新为
$$
\mathbf R_{t+k}=(d\mathbf M+\frac{(1-d)\mathbf E}{n})^k\mathbf R_t
$$
其中$d(0\leq d\leq1)$为阻尼因子，表示为节点以$d$的概率按照链接跳转，以$(1-d)$的概率任意跳转。

* $\mathbf E/n$的存在使得浏览者以$\frac{1}{n}$的概率跳转到任意页面，而不仅仅是通过超链接，有向图是连通图，因此随机游走过程是不可约的。
* 每个状态都有正概率转移到自身（$d\mathbf M+\frac{(1-d)\mathbf E}{n}$对角线元素为正），说明每个状态的周期是1，因此上述过程是非周期的。
* 由于上述过程是不可约和非周期的，且所有状态都有正概率转移到自身，这意味着从任何状态出发，平均返回到该状态的时间是有限的，因此上述过程是正常返还的。

### 3，数值求解

##### 迭代求解

利用迭代公式$\mathbf R_{t+k}=(d\mathbf M+\frac{(1-d)\mathbf E}{n})^k\mathbf R_t$迭代求解，直至$\mathbf R_t$趋于稳定，求解代码如下

```python
def pagerank_iter(M, d, tol=1.0e-6):
    n = M.shape[0]
    pagerank_vector = np.ones(n) / n
    base = (1 - d) / n * np.ones(n)
    while (True):
        new_rank = d * M @ pagerank_vector + base
        if np.linalg.norm(new_rank - pagerank_vector, ord=1) < tol:
            return new_rank
        pagerank_vector = new_rank
```

##### 代数求解

当经过足够长时间，达到马尔可夫链的平稳分布，$\mathbf R_t$收敛至$\mathbf R$时，满足
$$
\begin{align}
\mathbf R&=(d\mathbf M+\frac{(1-d)\mathbf E}{n})\mathbf R\\
&=d\mathbf {MR}+\frac{(1-d)\mathbf 1}{n}\\
&=(\mathbf I-d\mathbf M)^{-1}\frac{(1-d)\mathbf 1}{n}\\
\end{align}\\
$$
其中$\mathbf {ER}=[\sum_j\mathbf R_j]_{n}=[1]_{n}=\mathbf 1$为全1列向量，$\mathbf I$为$n\times n$单位矩。上述方法求解代码如下

```python
def pagerank_algebraic(M, d):
    I = np.eye(n)
    coeff_matrix = I - d * M
    b = np.ones(n) * (1 - d) / n
    pagerank_vector = scipy.linalg.solve(coeff_matrix, b)

    return pagerank_vector

```

##### MapReduce方式求解

根据迭代公式
$$
\mathbf R_{t+k}=(d\mathbf M+\frac{(1-d)\mathbf E}{n})^k\mathbf R_t
$$
可知对于$t+1$时刻的节点$i$，其`PageRank`值为
$$
\begin{align}
r_i^{t+1}&=d\mathbf M_{[i,:]}\mathbf R_t+\frac{(1-d)}{n}\\
&=d\sum_{j\in \mathbb m(i) }\frac{r^t_j}{L(\mathbb v(j))}+\frac{1-d}{n}
\end{align}\\
$$
其中$\mathbb m(i)$为存在指向节点$i$的节点的集合，$\mathbb v(i)$为节点$i$指向节点的集合。

在$t$时刻，节点$i$需要汇集在$t-1$时刻所有指向$i$的节点$j \in \mathbb m(i)$的`PageRank`值，用于更新$i$在$t$时刻的`PageRank`值，同时也需要向所有$i$  节点指向的节点$k \in \mathbb v(i)$分发$i$的`PageRank`值，用于$t+1$时刻节点$k$更新`PageRank`值。

整个计算过程可以拆分为`Map`与`Reduce`两个过程，利用分布式计算框架迭代更新，由于互联网网页数量是万亿级的数字，由于上述的迭代解法和代数解法需要在单机上运行，将无法处理万亿级数据，`MapReduce`方法则可以解决单机计算的性能瓶颈问题。

`Map`过程接收节点$j$的编号、`PageRank`值和节点$j$指向节点集合$v(j)$为输入，向可跳转节点$i$传递当前在$j$节点时，下一时刻游走到$i$节点的条件概率$P(i|j)$，用于更新$t+1$时刻$i$的`PageRank`值，完成节点$j$向可跳转节点的随机游走过程。同时向节点$j$本身传递$t$时刻的`PageRank`值和$v(j)$，用于判断$j$在$t+1$时刻更新后`PageRank`值是否收敛。

`Reduce`过程按照节点编号$i$汇集所有指向他的节点$j \in \mathbb m(i)$的条件概率$P(i|j)$，更新$t+1$时刻的`PageRank`值，并判断是否收敛，同时向下传递节点$i$编号，$\mathbb v(i)$和`PageRank`，用于下一轮`MapReduce`过程。

整个过程的伪代码如下

```python
Map(nid, node):
	yield nid, ('node', node)
	
	outlinks ,rank = unpack(node)
	for (outlink in outlinks):
		yield outlink, ('pagerank', rank / len(outlinks))

Reduce(nid, values):
	outlinks = []
	totalRank = 0
	oldRank = 0

	for (val in values):
		label, content = unpack(val)
		if label == 'node':
			outlinks = content[0]
			oldRank = content[1]
		else
			totalRank += content
	
	totalRank = (1 - d)/n + (d * totalRank)
	if check_err(oldRank, totalRank)> Thread:
		unconverted+=1
	yield nid, ('node', (outlinks,totalRank))
```

以下使用`mrjob`包完成`MapReduce`计算任务实现，完整代码可见[PageRank](https://github.com/HuangQiang97/PageRank)。

```python
import os
import shutil
import time

import numpy as np
from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol

from tools import check_err, load_data


class PageRank(MRJob):
    INPUT_PROTOCOL = JSONProtocol

    def configure_args(self):
        super(PageRank, self).configure_args()
        self.add_passthru_arg('--n', type=int)
        self.add_passthru_arg('--d', type=float)

    def mapper(self, nid, node):
        # 流向下一层更新pagerank处理
        yield nid, ('node', node)

        # 指向的其他节点, 当前节点pagerank
        adjacency_list, pagerank = node
        if len(adjacency_list) != 0:
            p = pagerank / len(adjacency_list)
            # 当前节点对他指向节点的贡献
            for adj in adjacency_list:
                yield adj, ('pagerank', p)

    def reducer(self, nid, values):
        # Initialize sum and node
        cur_sum = 0
        node = [[[], 0]]

        for val in values:
            label, content = val
            # 数据类型是node, 保存外链和pagerank值
            if label == 'node':
                node[0][0] = content[0]
                node[0][1] = content[1]
            # 数据类型是pagerank，计算所有指向当前节点vi的节点vj对vi的共享
            elif label == 'pagerank':
                cur_sum += content

        # 更新节点的PageRank值
        cur_sum = cur_sum * self.options.d + (1 - self.options.d) / self.options.n
        # 如果PageRank变化大于阈值，则视为未收敛
        if abs(cur_sum - node[0][1]) > 1e-9:
            self.increment_counter('nodes', 'unconverted_node_count', 1)

        node[0][1] = cur_sum
        node = tuple(*node)
        yield nid, node
```

### 4，验证

转移矩阵初始化为
$$
\mathbf M=\begin{bmatrix}
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & \frac{1}{2} & 0 & 0 & 0 & 0 & \frac{1}{2} \\
\frac{1}{2} & \frac{1}{2} & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & \frac{1}{3} & \frac{1}{3} & 0 & 0 & 0 & \frac{1}{3} & 0 \\
0 & \frac{1}{2} & 0 & 0 & \frac{1}{2} & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\end{bmatrix}
$$
阻滞因子$d=0.85$，状态分布向量初始化$\mathbf R_0=[1/8]_8$。

* 迭代法耗时`1.009`毫秒收敛，收敛值为

  `[0.14564, 0.18355, 0.04577, 0.29856, 0.02672, 0.01875, 0.02632, 0.03820]`
  迭代误差为`7.480E-7`。

* 代数求解耗时`5.991` 毫秒，收敛值为
  `[0.14564, 0.18355, 0.04577, 0.29856, 0.02672, 0.01875, 0.02632, 0.03820]`
  迭代误差为`6.9388E-17`

* `MapReduce`方式求解耗时`57.1386`秒，收敛值为
  `[0.14563, 0.18354, 0.04577, 0.29855, 0.02671, 0.01875, 0.02632, 0.03820]`
  迭代误差为`1.4726e-07`

迭代法和代数求解可以快速得到结果，并且代数求解可以获得最佳结果，但相较于其他方法，矩阵求逆时间复杂度较高，并且迭代法和代数求解只能在单机上运行，可运算数据规模受限。

`MapReduce`法求解时，由于每轮`Map`过程和 `Reduce`过程都涉及1次文件读写以及对象序列化和反序列化，且无法实现矩阵并行化计算，计算耗时最长，但是可以利用多机并行计算不受单机节点性能限制。

### 5，思考

通过上一小结发现`MapReduce`方式由于无法实现矩阵并行化计算，是性能较差的主要原因之一。观察公式
$$
\begin{align}
\mathbf R_{t+1}&=(d\mathbf M+\frac{(1-d)\mathbf E}{n})\mathbf R_t\\
&=d\mathbf {MR}_t+\frac{(1-d)\mathbf 1}{n}\\
\end{align}\\
$$
可以发现，迭代过程的矩阵运算可以分块进行。将上述表达式简化为$\mathbf Y=\mathbf{MR}+\mathbf B$，考虑将$\mathbf M$划分为$p\times q$个子阵，横向分割线下标为$[i_1,i_2,...,i_p]$，纵向分割线下标为$[j_1,j_2,...,j_q]$，同样将$\mathbf R$和$\mathbf B$划分为$q$个子阵，横向分割线下标为$[j_1,j_2,...,j_q]$，$\mathbf Y$划分为$p$个子阵列，横向分割线下标为$[i_1,i_2,...,i_p]$。$\mathbf Y$的第$i$个子阵可以表示为
$$
Y_{<i>} = \sum_{j=1}^{q} M_{<i,j>}R_{<j>} + B_{<i>},
$$
通过计算任务的拆分，将子阵分发到不同计算节点上，可以将大尺度矩阵运算切分为多个`MapReduce`小尺度矩阵运算子任务，将循环迭代计算替换为矩阵并行计算，缩短运算时间。以下为分块运算正确性验证，具体`MapReduce`任务待实现。

```python
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

Y = np.zeros_like(B) 

# 分块处理
for i in range(ki):
    start_i = block_i[i]
    end_i = block_i[i + 1]
    for j in range(kj):
        start_j = block_j[j]
        end_j = block_j[j + 1]
        Y[start_i:end_i] += M[start_i:end_i, start_j:end_j] @ PR[start_j:end_j]
Y += B  

# 3.51043408855362e-14
print(np.linalg.norm(Y - (M @ PR + B)))

```

### 6，代码结构

完整代码可见[PageRank](https://github.com/HuangQiang97/PageRank)：

* `tools.py`中的`generate_sparse_matrix`函数用于生成随机状态转移矩阵，方法将$\mathbf M$写入`data/input.txt`中，数据格式为"节点编号 [指向节点集合, 初始PageRank值]"。`check_err`函数用于计算当前$\mathbf R$在前后两次迭代间的差值。
* `matrix_pagerank.py`定义了问题的迭代法`pagerank_iter`和代数解法`pagerank_algebraic`。
* `mapredue_pagerank.py`定义了问题的`MapReduce`解法。
* `martix_mr_pagerank.py`验证了迭代方法中矩阵分块计算的正确性。
* `data`文件夹包含输入数据`input.txt`，以及`MapReduce`方法的输出结果。

### 7，引用

[^1]: [马尔可夫链]( https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE) 
[^2]: [PageRank算法详解](https://zhuanlan.zhihu.com/p/137561088)
[^3]: [搜索引擎设计](https://time.geekbang.org/column/article/493019) 
