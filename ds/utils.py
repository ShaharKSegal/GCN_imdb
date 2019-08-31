import os
import pickle

import pandas as pd

import config

from .cast_graph import CastGraph

pivot_df_names_cols = ['actor1_primaryName', 'actor2_primaryName',
                       'actor3_primaryName', 'director1_primaryName']
n_cast_per_movie = len(pivot_df_names_cols)
formatted_df_categories = [col.rstrip('_primaryName') for col in pivot_df_names_cols]


def save_pickle(pickle_path, obj):
    with open(os.path.join(config.data_path, f'{pickle_path}.pkl'), 'wb') as f:
        pickle.dump(obj, f)
    return


def load_pickle(pickle_path):
    with open(os.path.join(config.data_path, f'{pickle_path}.pkl'), 'rb') as f:
        ans = pickle.load(f)
    return ans


def get_pivot_df(path='all') -> pd.DataFrame:
    df = load_pickle(os.path.join(path, 'pivot_df'))
    # keep only names
    non_names_cols = ['runtimeMinutes', 'startYear', 'averageRating', 'numVotes']
    mask = (df.columns.str.contains('primaryName')) | (df.columns.isin(non_names_cols))
    pivot_df = df.iloc[:, mask]
    # filter out rows movies with "self"
    pivot_df = pivot_df[pivot_df.self1_primaryName.isna()]
    # filter columns with relevant jobs
    names_only_cols = [c for c in pivot_df_names_cols]
    pivot_df = pivot_df[[*non_names_cols, *names_only_cols]]
    # binarize rating
    pivot_df["label"] = pd.cut(pivot_df.averageRating, bins=[0.0, 7.0, 10.0], labels=False)
    pivot_df = pivot_df.dropna()
    return pivot_df


def get_formatted_df(path='all') -> pd.DataFrame:
    pivot_df = get_pivot_df()
    formatted_df = load_pickle(os.path.join(path, 'formatted_df'))
    formatted_df = formatted_df[formatted_df.tconst.isin(pivot_df.index)
                                & formatted_df.category.isin(formatted_df_categories)]
    return formatted_df


def get_people_df(path='') -> pd.DataFrame:
    return load_pickle(os.path.join(path, 'people_df'))


def get_cast_graph():
    return CastGraph(get_formatted_df(), get_people_df())
