import pandas as pd
from typing import List

import pandas as pd
import torch.utils.data

import config
from .utils import pivot_df_names_cols


class IMDBDataset(torch.utils.data.Dataset):
    label_column = 'label'
    feature_columns = ['runtimeMinutes', 'startYear', 'numVotes']

    def __init__(self, data):
        self._data = data

    @property
    def data(self) -> pd.DataFrame:
        """
        Data after preproccessing
        :return: data in pandas DataFrame format
        """
        return self._data

    @property
    def cast_columns(self) -> List[str]:
        return pivot_df_names_cols

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx) -> (torch.FloatTensor, torch.LongTensor, torch.LongTensor):
        row: pd.Series = self.data.iloc[idx]
        features = torch.from_numpy(row[self.feature_columns].values.astype(float)).float()
        label = torch.tensor(row[self.label_column]).long()
        cast_indices = [config.cast_graph.names[name] for name in row[self.cast_columns].to_list()]
        return features, label, torch.LongTensor(cast_indices)
