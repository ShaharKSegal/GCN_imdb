import pickle
import os

import numpy as np
import pandas as pd

from ds.utils import get_formatted_df
from ds.cast_graph import CastGraph

DATA_PATH = './data'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def save_pickle(pickle_path, obj):
    with open(os.path.join(DATA_PATH, f'{pickle_path}.pkl'), 'wb') as f:
        pickle.dump(obj, f)
    return


def load_pickle(pickle_path):
    with open(os.path.join(DATA_PATH, f'{pickle_path}.pkl'), 'rb') as f:
        ans = pickle.load(f)
    return ans


def generate_formatted_df():
    df = load_pickle('filtered_df')

    # numbering categories which have more than one person in them
    def add_number(series):
        return np.char.add(series.values.astype(str), np.arange(1, series.shape[0] + 1).astype(str))

    formatted_df = df.groupby(['tconst', 'category'], as_index=False).apply(lambda x: x.sort_values('ordering'))
    formatted_df.category = formatted_df.groupby(['tconst', 'category'], as_index=False).category.transform(add_number)
    formatted_df = formatted_df.reset_index(drop=True)
    save_pickle("formatted_df", formatted_df)
    return formatted_df


def generate_pivot_df():
    formatted_df = load_pickle('formatted_df')
    # pivot cast data
    movie_index = 'tconst'
    cast_column = 'category'
    cast_cols = ['nconst', 'primaryName', 'birthYear', 'deathYear', 'primaryProfession', 'knownForTitles', 'gender']
    movie_cols = [movie_index, 'primaryTitle', 'runtimeMinutes', 'startYear', 'averageRating', 'numVotes']

    cast_df = formatted_df.pivot(index=movie_index, columns=cast_column, values=cast_cols)
    cast_df.columns = ['_'.join(col[::-1]).strip() for col in cast_df.columns.values]
    movie_df = formatted_df[movie_cols].groupby(movie_index).first()
    pivot_df = movie_df.join(cast_df)
    save_pickle("pivot_df", pivot_df)
    return pivot_df


def generate_onehot_df():
    formatted_df = load_pickle('formatted_df')
    index = 'tconst'
    movie_cols = ['runtimeMinutes', 'startYear', 'averageRating', 'numVotes']
    df = pd.get_dummies(formatted_df.primaryName)
    names_cols = df.columns.to_list()
    df[index] = formatted_df[index]
    df[movie_cols] = formatted_df[movie_cols]

    movie_df: pd.DataFrame = df[[index, *movie_cols]].groupby(index).first()
    movie_df.astype(pd.SparseDtype("int", 0))
    ddf = df[[index, *names_cols]]
    cast_df = ddf.groupby(index).max().compute()
    save_pickle("movie_df", movie_df)
    save_pickle("cast_df", cast_df)
    save_pickle("onehot_df", movie_df.join(cast_df).to_sparse())


def filter_pivot():
    pivot_df = load_pickle('pivot_df')
    # keep only names
    non_names_cols = ['runtimeMinutes', 'startYear', 'averageRating', 'numVotes']
    mask = (pivot_df.columns.str.contains('primaryName')) | (pivot_df.columns.isin(non_names_cols))
    names_only_df = pivot_df.iloc[:, mask]
    # filter out rows movies with "self"
    names_only_df = names_only_df[names_only_df.self1_primaryName.isna()]
    # filter columns with relevant jobs
    names_only_cols = [c + '_primaryName' for c in
                       ['actor1', 'actor2', 'actor3', 'actor4', 'cinematographer1', 'composer1', 'director1', 'editor1',
                        'producer1', 'producer2', 'writer1', 'writer2']]
    names_only_df = names_only_df[[*non_names_cols, *names_only_cols]]
    # binarize rating
    names_only_df["label"] = pd.cut(names_only_df.averageRating, bins=[0.0, 7.0, 10.0], labels=False)
    names_only_df = names_only_df.drop(["averageRating", "runtimeMinutes", "startYear", "numVotes"], axis=1)

    return names_only_df


def get_cast_graph():
    pivot_df = filter_pivot()
    pivot_df = pivot_df[['actor1_primaryName', 'actor2_primaryName', 'actor3_primaryName',
                         'director1_primaryName', 'label']].dropna()
    formatted_df = load_pickle("formatted_df")
    formatted_df = formatted_df[formatted_df.tconst.isin(pivot_df.index)
                                & formatted_df.category.isin(['actor1', 'actor2', 'actor3', 'director1'])]
    return CastGraph(formatted_df)


def generate_people_df():
    formatted_df = get_formatted_df()
    people_df = formatted_df.groupby('primaryName').agg({
        'gender': 'first',
        'birthYear': 'first',
        'deathYear': 'first',
        'averageRating': 'mean',
        'startYear': 'sum',
        'tconst': 'count',
    })
    people_df['averageAge'] = people_df.startYear / people_df.tconst - people_df.birthYear
    people_df = people_df.drop(columns=['tconst', 'startYear'])
    people_df['gender'] = people_df.gender.replace('?', np.nan)
    people_df['gender'] = people_df.gender.factorize()[0]
    people_df = people_df.drop(columns='deathYear')

    people_df = (people_df - people_df.mean()) / people_df.std()
    return people_df
