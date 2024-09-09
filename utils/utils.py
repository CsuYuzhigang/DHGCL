import os
import dgl
import random

import pandas as pd
import torch as th
import math


def load_dataset(dataset, num_nodes, emb_size=128):
    hetero_graph_list = dgl.load_graphs(os.path.join('../data/', dataset, '{}.bin'.format(dataset)))[0]  # 异质图列表

    hetero_graph = hetero_graph_list[0]
    node_types = hetero_graph.ntypes
    # 为每种类型的节点生成节点特征
    node_representations = {}
    type_embeddings = create_type_embeddings(node_types, emb_size=len(node_types))  # 假设emb_size为16

    for node_type in node_types:
        # 获取每种类型节点的数量
        num_nodes = hetero_graph.number_of_nodes(node_type)

        # 生成位置编码
        feat = initial_feature(num_nodes, emb_size=emb_size)  # 假设emb_size为64
        # 获取类型嵌入并扩展维度以匹配位置编码
        type_embedding = type_embeddings[node_type].unsqueeze(0).repeat(num_nodes, 1)
        # 将类型嵌入与位置编码相加或相连接
        node_representations[node_type] = feat
    return hetero_graph_list, node_representations


def split_dataset(dataset, num_nodes):  # 划分数据集
    df_labels = pd.read_csv(os.path.join('../data/', dataset, '{}_label.txt'.format(dataset)), sep=' ', names=['id', 'label'])
    num_label_nodes = df_labels.shape[0]  # 带标签节点数
    labels = th.full((num_nodes,), -1).cuda()  # 标签
    for index, row in df_labels.iterrows():
        labels[row['id']] = row['label']
    n_classes = df_labels.label.nunique()  # 类别数
    print('classes number: {}'.format(n_classes))

    train_mask = th.full((num_nodes,), False)  # 训练集
    val_mask = th.full((num_nodes,), False)  # 验证集
    test_mask = th.full((num_nodes,), False)  # 测试集

    random.seed(2024)
    train_mask_index, val_mask_index, test_mask_index = th.LongTensor([]), th.LongTensor([]), th.LongTensor([])
    # 对每个类别均匀划分
    for i in range(n_classes):
        index = [j for j in df_labels[df_labels.label == i].id.tolist()]
        random.shuffle(index)  # 打乱
        # 训练集取 60%
        train_mask_index = th.cat((train_mask_index, th.LongTensor(index[:int(len(index) * 0.6)])), 0)
        # 验证集取 20%
        val_mask_index = th.cat((val_mask_index, th.LongTensor(index[int(len(index) * 0.6):int(len(index) * 0.8)])), 0)
        # 测试集取 20%
        test_mask_index = th.cat((test_mask_index, th.LongTensor(index[int(len(index) * 0.8):])), 0)

    train_mask.index_fill_(0, train_mask_index, True).cuda()  # 训练集
    val_mask.index_fill_(0, val_mask_index, True).cuda()  # 验证集
    test_mask.index_fill_(0, test_mask_index, True).cuda()  # 测试集

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()  # 训练集
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()  # 验证集
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()  # 测试集

    return train_idx, val_idx, test_idx, labels


def create_type_embeddings(node_types, emb_size):
    # 为每种类型的节点创建一个唯一的嵌入向量
    type_embeddings = th.eye(len(node_types), emb_size)
    type_embedding_dict = {node_type: type_embeddings[i] for i, node_type in enumerate(node_types)}
    return type_embedding_dict


def initial_feature(max_len, emb_size):  # 编码
    import torch.nn as thnn
    emb = thnn.Parameter(
        th.Tensor(max_len, emb_size), requires_grad=True
    )
    thnn.init.xavier_uniform_(emb)

    return emb


def sampling_layer(snapshots, views, strategy='random'):  # 采样层
    samples = []  # 采样结果
    random.seed(2024)

    if strategy == 'random':  # 随机采样
        samples = random.sample(range(0, snapshots), views)  # 随机采取 views 个样本
    elif strategy == 'sequential':  # 顺序采样
        samples = random.sample(range(0, snapshots - views + 1), 1)  # 随机采取 1 个样本
        start = samples[0]
        for i in range(1, views):
            samples.append(start + i)  # 按顺序取剩下的样本

    return samples
