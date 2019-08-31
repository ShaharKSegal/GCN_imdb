import torch
import torch.nn as nn
import dgl

from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self, graph: dgl.DGLGraph, in_feats: int, n_hidden: int, n_outs: int, n_layers: int, activation):
        super(GCN, self).__init__()
        self.graph = graph
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_outs))

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(self.graph, h)
        return h
