import os

import torch.utils.data

import config

from .dataset import IMDBDataset
from .utils import get_pivot_df


def get_loaders():
    def create_loader(data_set, train_mode=False):
        if train_mode:  # add randomized sampling potentially with weights
            sampler = torch.utils.data.RandomSampler(data_set, replacement=True)
        else:
            sampler = torch.utils.data.SequentialSampler(data_set)
        return torch.utils.data.DataLoader(dataset=data_set, sampler=sampler, batch_size=config.batch_size)

    ds_train = IMDBDataset(get_pivot_df('train'))
    ds_eval = IMDBDataset(get_pivot_df('eval'))
    ds_test = IMDBDataset(get_pivot_df('test'))
    train_loader = create_loader(ds_train, True)
    eval_loader = create_loader(ds_eval)
    test_loader = create_loader(ds_test)
    return train_loader, eval_loader, test_loader
