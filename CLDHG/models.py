import torch.nn as thnn
import torch as th
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn.pytorch import GraphConv, HeteroGraphConv, HeteroLinear, SAGEConv


class LogReg(thnn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = thnn.Linear(hid_dim, out_dim)  # 线性层

    def forward(self, x):
        ret = self.fc(x)  # 前向传播
        return ret


class MLPLinear(thnn.Module):  # 线性层
    def __init__(self, node_types, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear1 = HeteroLinear({node: in_dim for node in node_types}, out_dim)  # 线性层 1
        self.linear2 = HeteroLinear({node: out_dim for node in node_types}, out_dim)  # 线性层 2
        self.act = thnn.LeakyReLU(0.2)  # LeakyReLU 激活函数
        self.reset_parameters()  # 初始化参数

    def reset_parameters(self):  # 初始化参数
        for param in self.linear1.parameters():
            if param.dim() > 1:
                thnn.init.xavier_uniform_(param)
        for param in self.linear2.parameters():
            if param.dim() > 1:
                thnn.init.xavier_uniform_(param)

    def forward(self, x):  # 前向传播
        # 线性变换
        x = self.linear1(x)
        # 对字典特征进行非线性处理
        x = {key: self.act(F.normalize(tensor, p=2, dim=1)) for key, tensor in x.items()}
        # 线性变换
        x = self.linear2(x)
        # 对字典特征进行非线性处理
        x = {key: self.act(F.normalize(tensor, p=2, dim=1)) for key, tensor in x.items()}

        return x


class HeteroGraphConvModel(thnn.Module):
    def __init__(self,
                 edge_types,
                 node_types,
                 in_feats,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 norm,
                 activation,
                 aggregate='sum',
                 readout='max'):
        super(HeteroGraphConvModel, self).__init__()
        self.edge_types = edge_types
        self.node_types = node_types
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.norm = norm
        self.activation = activation
        self.aggregate = aggregate
        self.readout = readout
        self.dropout = 0.5
        self.layers = thnn.ModuleList()
        #
        # build multiple layers
        self.layers.append(HeteroGraphConv(
            {edge: SAGEConv(self.in_feats, self.hidden_dim, activation=self.activation, aggregator_type='mean'
                            ) for edge in self.edge_types},
            aggregate=self.aggregate))  # 第一个异质图卷积层
        for i in range(1, (self.n_layers - 1)):
            self.layers.append(HeteroGraphConv(
                {edge: SAGEConv(self.hidden_dim, self.hidden_dim, activation=self.activation, aggregator_type='mean')
                 for edge in self.edge_types},
                aggregate=self.aggregate))  # 中间的异质图卷积层
        self.layers.append(HeteroGraphConv(
            mods={edge: SAGEConv(self.hidden_dim, self.output_dim, aggregator_type='mean'
                                 ) for edge in self.edge_types},
            aggregate=self.aggregate))  # 最后一个异质图卷积层

        self.linear = HeteroLinear({node: self.output_dim for node in self.node_types},
                                   self.output_dim)  # 添加一个线性层，用于将最后的特征进行线性变换

        self.act = thnn.LeakyReLU(0.2)  # LeakyReLU 激活函数

    def forward(self, G):  # 前向传播
        input_dict = {ntype: G.nodes[ntype].data['feat'] for ntype in G.ntypes}
        # Todo
        for i in range(self.n_layers):
            h_dict = self.layers[i](G, input_dict)
            input_dict = h_dict
        # 线性变换
        h_dict = self.linear(input_dict)
        # 对字典特征进行非线性处理
        h_dict = {key: self.act(F.normalize(tensor, p=2, dim=1)) for key, tensor in h_dict.items()}

        return h_dict
