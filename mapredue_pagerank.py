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
        if abs(cur_sum - node[0][1]) > 1e-7:
            self.increment_counter('nodes', 'unconverted_node_count', 1)

        node[0][1] = cur_sum
        node = tuple(*node)

        yield nid, node


if __name__ == '__main__':
    input_file = './data/input.txt'
    output_dir = './data/output/{}/'

    iteration = 0
    d = 0.85
    with open(input_file, 'r') as f:
        n = len(f.readlines())

    start = time.time_ns()
    while True:
        print('iteration: {}'.format(iteration + 1))
        output_path = '--output-dir=' + output_dir.format(iteration)

        # 开启新一轮任务
        if iteration == 0:
            job = PageRank([input_file, output_path, '--n=%d' % n, '--d=%f' % d])
        else:
            job = PageRank([output_dir.format(iteration - 1) + '*', output_path, '--n=%d' % n, '--d=%f' % d])

        # 运行任务
        with job.make_runner() as runner:
            runner.run()
            unconverted_node_count = 0
            # 当所有节点都收敛后停止运行
            for val in runner.counters():
                try:
                    unconverted_node_count += val['nodes']['unconverted_node_count']
                except KeyError:
                    pass

            print(unconverted_node_count, "unconverted_node_count")
            if unconverted_node_count == 0:
                break

        iteration += 1
    # 耗时
    print('time cost: %e s' % ((time.time_ns() - start) / 1e9))
    # 收集结果
    ranks = np.zeros((n,))
    for root, dirs, files in os.walk(output_dir.format(iteration)):
        for name in files:
            file_path = os.path.join(root, name)
            with open(file_path, 'r', encoding='utf8') as f:
                for line in f.readlines():
                    key, value = line.split('\t')
                    rank = float(value.split(',')[-1][1:-2])
                    ranks[int(key) - 1] = rank
    with open('./data/output.txt', 'w', encoding='utf8') as f:
        f.write(repr(ranks.tolist()))
    shutil.rmtree('./data/output')
    print('pagerank:%s' % ranks)
    # 计算误差
    print('err: %e' % check_err(load_data(), ranks, d))
