import torch
import torch.nn as nn

import config

from ds import n_cast_per_movie, IMDBDataset


def simple_dnn(n_cast_feat: int, n_hidden: int, n_movie_feats: int, n_cast_num: int):
    return nn.Sequential(
        nn.Linear(n_cast_feat * n_cast_num + n_movie_feats, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, 2)
    )


class SimpleNN(nn.Module):
    def __init__(self, n_hidden: int, n_movie_feats: int = None, n_cast_num: int = None):
        super(SimpleNN, self).__init__()
        if n_movie_feats is None:
            n_movie_feats = len(IMDBDataset.feature_columns)
        if n_cast_num is None:
            n_cast_num = n_cast_per_movie
        n_graph_feats = config.cast_graph.graph_features.shape[1]
        self.fc = simple_dnn(n_graph_feats, n_hidden, n_movie_feats, n_cast_num)

    def forward(self, graph_features, movie_features, cast_indices):
        # slice cast features to act like embedding on a batch
        cast_features = graph_features[cast_indices.flatten()].view(cast_indices.shape[0], -1)
        # concat with movie features
        fc_features = torch.cat((cast_features, movie_features), 1)
        out = self.fc(fc_features)
        return out
