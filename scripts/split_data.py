import os

import numpy as np
import pandas as pd

import config

from ds.utils import load_pickle, save_pickle, get_formatted_df, get_pivot_df

test_size = 0.2
eval_size = 0.2

formatted_df: pd.DataFrame = load_pickle('formatted_df')
pivot_df: pd.DataFrame = load_pickle('pivot_df')


def split_data(piv_df, format_df, split_size=0.2):
    movies = piv_df.index
    n_movies = movies.shape[0]
    train_movies_indices = np.random.permutation(np.arange(n_movies))[:int((1 - split_size) * n_movies)]
    train_movies = movies[train_movies_indices]

    first_piv_df = piv_df[piv_df.index.isin(train_movies)]
    first_format_df = format_df[format_df.tconst.isin(train_movies)]
    second_piv_df = piv_df[~piv_df.index.isin(train_movies)]
    second_format_df = format_df[~format_df.tconst.isin(train_movies)]

    return first_piv_df, first_format_df, second_piv_df, second_format_df


train_eval_pivot_df, train_eval_formatted_df, test_pivot_df, test_formatted_df = split_data(pivot_df, formatted_df,
                                                                                            test_size)
train_pivot_df, train_formatted_df, eval_pivot_df, eval_formatted_df = split_data(train_eval_pivot_df,
                                                                                  train_eval_formatted_df, test_size)

save_pickle(os.path.join('train', 'pivot_df'), train_pivot_df)
save_pickle(os.path.join('train', 'formatted_df'), train_formatted_df)
save_pickle(os.path.join('eval', 'pivot_df'), eval_pivot_df)
save_pickle(os.path.join('eval', 'formatted_df'), eval_formatted_df)
save_pickle(os.path.join('test', 'pivot_df'), test_pivot_df)
save_pickle(os.path.join('test', 'formatted_df'), test_formatted_df)
