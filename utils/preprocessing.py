import os
import time
from datetime import datetime

import pandas as pd

from data_processing import get_dblp, get_aminer


# Math-Overflow 预处理
def preprocessing_for_math_overflow():
    file_paths = [os.path.join('./data', 'MathOverflow', 'a2q.txt'),
                  os.path.join('./data', 'MathOverflow', 'c2a.txt'),
                  os.path.join('./data', 'MathOverflow', 'c2q.txt')]  # 文件路径
    edge_types = ['a2q', 'c2a', 'c2q']  # 边的种类
    for file_path, edge_type in zip(file_paths, edge_types):
        # 读取文件的所有行
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        # 在每一行的末尾添加指定的单词
        modified_lines = [line.strip() + ' ' + edge_type for line in lines]
        # 加入换行符，但最后一行不加
        modified_content = '\n'.join(modified_lines)
        # 将修改后的行写回到原文件中
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(modified_content)
    # 合并三个文件
    output_file = os.path.join('./data', 'MathOverflow', 'MathOverflow.txt')
    # 打开并读取第一个文件
    with open(file_paths[0], 'r', encoding='utf-8') as f1:
        content1 = f1.read()

    # 打开并读取第二个文件
    with open(file_paths[1], 'r', encoding='utf-8') as f2:
        content2 = f2.read()

    # 打开并读取第三个文件
    with open(file_paths[2], 'r', encoding='utf-8') as f3:
        content3 = f3.read()

    # 将所有内容写入新的目标文件
    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write(content1)
        out_file.write('\n')  # 添加换行符以分隔文件内容
        out_file.write(content2)
        out_file.write('\n')  # 添加换行符以分隔文件内容
        out_file.write(content3)


# EComm 预处理
def preprocessing_for_ecomm():
    file_path = os.path.join('./data', 'EComm', 'EComm.txt')
    df = pd.read_csv(file_path, delimiter='\t', header=None)  # 读取数据
    df[2], df[3] = df[3], df[2]  # 交换两列
    edge_type_map = {1: 'click', 2: 'buy', 3: 'a2c', 4: 'a2f'}  # 边的种类
    df[3] = df[3].map(edge_type_map)  # 边的种类映射
    df.to_csv(file_path, sep=' ', index=False, header=False)  # 保存


# Yelp 预处理
def preprocessing_for_yelp():
    # 处理 label 文件
    file_path = os.path.join('../data', 'Yelp', 'Yelp_label_old.txt')
    df = pd.read_csv(file_path, delimiter=' ', header=None)  # 读取数据
    df[0] = df[0] - 1  # id 从 0 开始
    df.to_csv(os.path.join('../data', 'Yelp', 'Yelp_label.txt'), sep=' ', index=False, header=False)  # 保存

    # 处理数据文件
    file_path = os.path.join('../data', 'Yelp', 'Yelp_old.txt')
    df = pd.read_csv(file_path, delimiter=' ', header=None)  # 读取数据
    df.columns = ['user', 'item', 'rating', 'years', 'time']
    df['user'] = df['user'] - 1  # id
    df['item'] = df['item'] - 1
    df = df[['item', 'user', 'years', 'time']]  # 交换两列
    df['Time'] = (df['years'] + ' ' + df['time']).apply(
        lambda x: int(time.mktime(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timetuple())))  # 时间戳转换为整型
    df['edge'] = 'buy'
    df.drop(['years', 'time'], axis=1, inplace=True)
    df.to_csv(os.path.join('../data', 'Yelp', 'Yelp.txt'), sep=' ', index=False, header=False)  # 保存


# DBLP 预处理
def preprocessing_for_dblp():
    # 处理数据文件
    file_path = os.path.join('../data', 'DBLP', 'DBLP_old.txt')
    df = pd.read_csv(file_path, delimiter=' ', header=None)  # 读取数据
    # 处理节点序号, 使之连续
    author_map = {}  # 作者序号映射 map
    paper_map = {}  # 论文序号映射 map
    author_num = 0  # 作者初始序号
    paper_num = 0  # 论文初始序号
    for index, row in df.iterrows():
        if row[0] not in author_map:
            author_map[row[0]] = author_num
            author_num += 1
        if row[1] not in paper_map:
            paper_map[row[1]] = paper_num
            paper_num += 1
    df[0] = df[0].map(author_map)
    df[1] = df[1].map(paper_map)
    df[3] = 'write'
    df.to_csv(os.path.join('../data', 'DBLP', 'DBLP.txt'), sep=' ', index=False, header=False)  # 保存

    # 处理 label 文件
    file_path = os.path.join('../data', 'DBLP', 'DBLP_label_old.txt')
    df = pd.read_csv(file_path, delimiter=' ', header=None)  # 读取数据
    # 处理节点序号
    df[0].map(author_map)
    df.dropna(inplace=True)  # 删除空行
    df.sort_values(by=[0], inplace=True)  # 排序
    df.to_csv(os.path.join('../data', 'DBLP', 'DBLP_label.txt'), sep=' ', index=False, header=False)  # 保存


# Aminer 预处理
def preprocessing_for_aminer():
    # 处理数据文件
    file_path = os.path.join('../data', 'Aminer', 'Aminer_old.txt')
    df = pd.read_csv(file_path, delimiter=' ', header=None)  # 读取数据
    # 处理节点序号, 使之连续
    author_map = {}  # 作者序号映射 map
    paper_map = {}  # 论文序号映射 map
    author_num = 0  # 作者初始序号
    paper_num = 0  # 论文初始序号
    for index, row in df.iterrows():
        if row[0] not in author_map:
            author_map[row[0]] = author_num
            author_num += 1
        if row[1] not in paper_map:
            paper_map[row[1]] = paper_num
            paper_num += 1
    df[0] = df[0].map(author_map)
    df[1] = df[1].map(paper_map)
    df[3] = 'write'
    df.to_csv(os.path.join('../data', 'Aminer', 'Aminer.txt'), sep=' ', index=False, header=False)  # 保存

    # 处理 label 文件
    file_path = os.path.join('../data', 'Aminer', 'Aminer_label_old.txt')
    df = pd.read_csv(file_path, delimiter=' ', header=None)  # 读取数据
    # 处理节点序号
    df[0].map(author_map)
    df.dropna(inplace=True)  # 删除空行
    df.sort_values(by=[0], inplace=True)  # 排序
    df.to_csv(os.path.join('../data', 'Aminer', 'Aminer_label.txt'), sep=' ', index=False, header=False)  # 保存


if __name__ == '__main__':
    get_aminer()
