# encoding: utf-8
import argparse
import numpy as np
import torch as th
import torch.nn as thnn
import torch.nn.functional as F
import dgl.function as fn
import torch.optim as optim
from functools import reduce
import random
import dgl
import wandb
import copy
from dgl.dataloading import MultiLayerNeighborSampler, MultiLayerFullNeighborSampler
from dgl.dataloading import DataLoader
# from dgl.dataloading import NodeDataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import f1_score

from models import MLPLinear, LogReg, HeteroGraphConvModel
from utils.data_processing import get_twitter, get_math_overflow, get_ecomm, get_yelp
from utils.utils import load_dataset, split_dataset, sampling_layer

random.seed(2024)


def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        emb[ntype] = node_embed[ntype][nid]
    return emb


def train(dataset, hidden_dim, n_layers, output_dim, fanouts, snapshots, views, strategy, readout, batch_size,
          dataloader_size, num_workers, epochs, GPU):
    device_id = GPU
    # 数据集处理函数映射字典
    data_processing_dict = {'Twitter': get_twitter, 'MathOverflow': get_math_overflow, 'EComm': get_ecomm,
                            'Yelp': get_yelp}

    edge_types, node_types_dict = data_processing_dict.get(dataset)()  # 数据集预处理
    temporal_hetero_graph_list, node_feat = load_dataset(dataset, sum(list(node_types_dict.values())))  # 加载数据集

    # 给每个时间图 传入节点特征
    for ntype in temporal_hetero_graph_list[0].ntypes:
        for sg_id in range(snapshots):
            temporal_hetero_graph_list[sg_id].nodes[ntype].data['feat'] = node_feat[ntype]

    sampler = MultiLayerNeighborSampler(fanouts)  # 初始化采样器
    in_feats = node_feat[list(node_feat.keys())[0]].shape[1]  # 输入特征维度 默认不同类型节点的特征维度相同

    model = HeteroGraphConvModel(edge_types, list(node_types_dict.keys()), in_feats, hidden_dim, output_dim, n_layers,
                                 norm='right', activation=F.relu, aggregate='max', readout=readout)  # 模型

    model = model.to(device_id)
    projection_model = MLPLinear(list(node_types_dict.keys()), output_dim, output_dim).to(device_id)

    loss_fn = thnn.CrossEntropyLoss().to(device_id)  # 交叉熵损失函数

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 4e-3, 'weight_decay': 5e-4},
        {'params': projection_model.parameters(), 'lr': 4e-3, 'weight_decay': 5e-4}
    ])  # 使用 Adam 优化器, 配置模型参数的学习率和权重衰减

    best_loss = th.tensor([float('inf')]).to(device_id)  # 初始化最优损失为无穷大
    best_model = copy.deepcopy(model)  # 保存模型初始副本
    print('Plan to train {} epoches \n'.format(epochs))

    for epoch in range(epochs):  # 多轮训练
        # mini-batch for training
        model.train()
        projection_model.train()
        train_dataloader_list = []  # 时间子图, 节点 id, 训练数据加载器

        # 确认所关心节点类型，并进行对比学习；
        target_num_nodes = temporal_hetero_graph_list[0].number_of_nodes(list(node_feat.keys())[0])
        train_nid_per_gpu = random.sample((list(range(target_num_nodes))), batch_size)
        random.shuffle(train_nid_per_gpu)
        train_nid_per_gpu = th.tensor(train_nid_per_gpu)

        for sg_id in range(snapshots):  # views
            train_dataloader = DataLoader(temporal_hetero_graph_list[sg_id],
                                          {'item': train_nid_per_gpu},
                                          sampler,
                                          batch_size=train_nid_per_gpu.shape[0],
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=num_workers,
                                          )
            train_dataloader_list.append(train_dataloader)

        seeds_emb = th.tensor([]).to(device_id)
        for train_dataloader in train_dataloader_list:
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                # forward
                blocks = [block.to(device_id) for block in blocks]
                batch_inputs = extract_embed(node_feat, input_nodes)
                batch_inputs = {k: e.cuda().to(device_id) for k, e in batch_inputs.items()}
                # metric and loss
                train_batch_logits = model(blocks, batch_inputs)
                train_batch_logits = projection_model(train_batch_logits)

                logits = th.tensor([]).to(device_id)
                for _, tensor in train_batch_logits.items():
                    logits = th.cat([logits, tensor], dim=0)  # 将字典中的张量拼接
                seeds_emb = th.cat([seeds_emb, logits.unsqueeze(0)], dim=0)
                # torch.Size([views: 4, batchsize: 256, output_dim: 64])
        train_contrastive_loss = th.tensor([0]).to(device_id)
        for idx in range(seeds_emb.shape[0] - 1):
            z1 = seeds_emb[idx]
            # z1.shape = torch.size([batchszie:256, output_dim: 64])
            z2 = seeds_emb[idx + 1]
            # z2.shape = torch.size([batchszie:256, output_dim: 64])
            pred1 = th.mm(z1, z2.T).to(device_id)
            # pred1.shape = torch.size([batchsize: 256, batchsize: 256])
            pred2 = th.mm(z2, z1.T).to(device_id)

            labels = th.arange(pred1.shape[0]).to(device_id)
            # tensor([  0,   1,   2,   3,   4,   5,   6,  batchsize-1], device='cuda:0')
            train_contrastive_loss = train_contrastive_loss + (
                    loss_fn(pred1 / 0.07, labels) + loss_fn(pred2 / 0.07, labels)) / 2

        optimizer.zero_grad()
        train_contrastive_loss.backward()
        optimizer.step()

        if train_contrastive_loss < best_loss:
            best_loss = train_contrastive_loss
            best_model = copy.deepcopy(model)
        print("Epoch {:05d} | Loss {:.4f} ".format(epoch, float(train_contrastive_loss)))

    sampler = MultiLayerFullNeighborSampler(2)

    # TODO: test
    # graph = dgl.to_simple(graph)
    # graph = dgl.to_bidirected(graph, copy_ndata=True)
    # test_dataloader = DataLoader(graph,
    #                              graph.nodes(),
    #                              sampler,
    #                              batch_size=dataloader_size,
    #                              shuffle=False,
    #                              drop_last=False,
    #                              num_workers=num_workers,
    #                              )

    best_model.eval()
    '''
    for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
        batch_inputs = node_feat[input_nodes].to(device_id)  # 加载子张量至指定的设备
        blocks = [block.to(device_id) for block in blocks]
        test_batch_logits = best_model(blocks, batch_inputs)

        batch_embeddings = test_batch_logits.detach()
        if step == 0:
            embeddings = batch_embeddings
        else:
            embeddings = th.cat((embeddings, batch_embeddings), axis=0)

    # Linear Evaluation
    train_idx, val_idx, test_idx = split_dataset(num_nodes_list)
    train_embs = embeddings[train_idx].to(device_id)
    val_embs = embeddings[val_idx].to(device_id)
    test_embs = embeddings[test_idx].to(device_id)

    label = labels.to(device_id)

    train_labels = th.tensor(label[train_idx])
    val_labels = th.tensor(label[val_idx])
    test_labels = th.tensor(label[test_idx])

    micros, weights = [], []
    for _ in range(5):
        logreg = LogReg(train_embs.shape[1], output_dim)
        logreg = logreg.to(device_id)
        loss_fn = thnn.CrossEntropyLoss()
        opt = th.optim.Adam(logreg.parameters(), lr=1e-2, weight_decay=1e-4)

        best_val_acc, eval_micro, eval_weight = 0, 0, 0
        for epoch in range(2000):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = th.argmax(logits, dim=1)
            train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

            logreg.eval()
            with th.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = th.argmax(val_logits, dim=1)
                test_preds = th.argmax(test_logits, dim=1)

                val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]

                ys = test_labels.cpu().numpy()
                indices = test_preds.cpu().numpy()
                test_micro = th.tensor(f1_score(ys, indices, average='micro'))
                test_weight = th.tensor(f1_score(ys, indices, average='weighted'))

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    if (test_micro + test_weight) >= (eval_micro + eval_weight):
                        eval_micro = test_micro
                        eval_weight = test_weight

                print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_micro:{:4f}, test_weight:{:4f}'.format(epoch,
                                                                                                              train_acc,
                                                                                                              val_acc,
                                                                                                              test_micro,
                                                                                                              test_weight))
        micros.append(eval_micro)
        weights.append(eval_weight)

    micros, weights = th.stack(micros), th.stack(weights)
    print('Linear evaluation Accuracy:{:.4f}, Weighted-F1={:.4f}'.format(micros.mean().item(), weights.mean().item()))
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DHGCL')
    parser.add_argument('--dataset', type=str, help="Name of the dataset.")
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--output_dim', type=int, required=True)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument("--fanout", type=str, required=True, help="fanout numbers", default='20,20')
    parser.add_argument('--snapshots', type=int, default=4)
    parser.add_argument('--views', type=int, default=2)
    parser.add_argument('--strategy', type=str, default='random')
    parser.add_argument('--readout', type=str, default='max')
    parser.add_argument('--batch_size', type=int, default=800)
    parser.add_argument('--dataloader_size', type=int, default=4096)
    parser.add_argument('--GPU', type=int, required=True)
    parser.add_argument('--num_workers_per_gpu', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    # parse arguments
    DATASET = args.dataset
    HID_DIM = args.hidden_dim
    OUTPUT_DIM = args.output_dim
    N_LAYERS = args.n_layers
    FANOUTS = [int(i) for i in args.fanout.split(',')]
    SNAPSHOTS = args.snapshots
    VIEWS = args.views
    STRATEGY = args.strategy
    READOUT = args.readout
    BATCH_SIZE = args.batch_size
    DATALOADER_SIZE = args.dataloader_size
    GPU = args.GPU
    WORKERS = args.num_workers_per_gpu
    EPOCHS = args.epochs

    # output arguments for logging
    print('Dataset: {}'.format(DATASET))
    print('Hidden dimensions: {}'.format(HID_DIM))
    print('number of hidden layers: {}'.format(N_LAYERS))
    print('Fanout list: {}'.format(FANOUTS))
    print('Batch size: {}'.format(BATCH_SIZE))
    print('GPU: {}'.format(GPU))
    print('Number of workers per GPU: {}'.format(WORKERS))
    print('Max number of epochs: {}'.format(EPOCHS))

    train(dataset=DATASET, hidden_dim=HID_DIM, n_layers=N_LAYERS, output_dim=OUTPUT_DIM,
          fanouts=FANOUTS, snapshots=SNAPSHOTS, views=VIEWS, strategy=STRATEGY, readout=READOUT,
          batch_size=BATCH_SIZE, dataloader_size=DATALOADER_SIZE, num_workers=WORKERS, epochs=EPOCHS, GPU=GPU)
