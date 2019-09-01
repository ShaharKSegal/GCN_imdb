import itertools
from collections import Counter

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse import dok_matrix


class CastGraph:
    df_col = 'primaryName'

    def __init__(self, samples: pd.DataFrame, samples_features: pd.DataFrame):
        self.samples = samples
        sorted_names = np.sort(self.samples[self.df_col].unique())
        samples_features = samples_features.sort_index()
        self.names = {name: i for i, name in enumerate(sorted_names)}
        self.rnames = {v: k for k, v in self.names.items()}
        self.empirical_single_marginals = Counter()
        self.empirical_pair_marginals = Counter()
        self.adj_matrix = dok_matrix((len(self.names), len(self.names)))

        self.process_samples()

        for (label1, label2), count in self.empirical_pair_marginals.items():
            self.adj_matrix[label1, label2] = self.adj_matrix[label2, label1] = count

        for label, count in self.empirical_single_marginals.items():
            self.adj_matrix[label, label] = count

        self.nx_graph: nx.Graph = nx.from_scipy_sparse_matrix(self.adj_matrix)
        nx.relabel_nodes(self.nx_graph, self.rnames)
        self.dgl_graph = dgl.DGLGraph(self.nx_graph)
        self.graph_features = samples_features.values

    def process_samples(self):
        image_groups = self.samples.groupby("tconst")
        for _, group in image_groups:
            labels = group[self.df_col]
            for label in labels.unique():
                label = self.names[label]
                self.empirical_single_marginals[label] += 1
            for label1, label2 in itertools.combinations(labels.values, 2):
                label1, label2 = self.names[label1], self.names[label2]
                if label1 == label2:
                    continue
                elif label1 < label2:
                    label1, label2 = label2, label1
                self.empirical_pair_marginals[(label1, label2)] += 1
