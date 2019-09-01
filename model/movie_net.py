import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

import config

from .gcn import GCN
from .simple import simple_dnn
from ds import n_cast_per_movie, IMDBDataset


class MovieNet(nn.Module):
    def __init__(self, n_gcn_hidden: int, n_gcn_out: int, n_gcn_layers: int, n_fc_hidden: int,
                 n_movie_feats: int = None, n_fc_cast_num: int = None):
        super(MovieNet, self).__init__()
        graph: dgl.DGLGraph = config.cast_graph.dgl_graph
        n_graph_feats: int = config.cast_graph.graph_features.shape[1]

        self.gcn = GCN(graph, n_graph_feats, n_gcn_hidden, n_gcn_out, n_gcn_layers, F.relu)
        if n_movie_feats is None:
            n_movie_feats = len(IMDBDataset.feature_columns)
        if n_fc_cast_num is None:
            n_fc_cast_num = n_cast_per_movie
        self.fc = simple_dnn(n_gcn_out, n_fc_hidden, n_movie_feats, n_fc_cast_num)

    def forward(self, graph_features, movie_features, cast_indices):
        # gcn embedding
        embed_mat = self.gcn(graph_features)

        # slice gcn output to act like embedding on a batch
        cast_features = embed_mat[cast_indices.flatten()].view(cast_indices.shape[0], -1)
        # concat with movie features
        fc_features = torch.cat((cast_features, movie_features), 1)
        out = self.fc(fc_features)
        return out
