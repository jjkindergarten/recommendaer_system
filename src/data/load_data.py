from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
import pandas as pd
import numpy as np
import surprise
from src.utilits.share import mse, mae
from src.collaborative_filtering.SVD_surprise import *

# Load the movielens-100k dataset (download it if needed).
def load_movielens_100k():
    data = pd.read_csv('data/MovieLens/ml-100k/u.data', sep = '\t', header=None, names = ['user', 'movie', 'rating', 'timestamp'])
    data = data.drop(['timestamp'], axis=1)

    data["user"] -= 1
    data["movie"] -= 1
    for col in ("user", "movie"):
        data[col] = data[col].astype(np.int32)
    data["rating"] = data["rating"].astype(np.float32)

    split_ratio = 0.75
    rows = len(data)
    data = data.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * split_ratio)
    data_train_db = data[:split_index]
    data_test = data[split_index:].reset_index(drop=True)

    data_train = surprise.Dataset.load_from_df(data_train_db, reader=surprise.Reader('ml-100k')).build_full_trainset()
    return data, data_train, data_test, data_train_db

# most popular item and most freq use user
def load_popular_sub_data(data, n_user, n_item):
    """
    
    :param data: 
    :param n_user: 
    :param n_item: 
    :return: 
    """
    user_freq = data['user'].value_counts(ascending = False)[:n_user]
    freq_user = list(user_freq.index)
    data = data[data.user.isin(freq_user)]

    item_freq = data['movie'].value_counts(ascending = False)[:n_item]
    freq_item = list(item_freq.index)
    data = data[data.movie.isin(freq_item)].reset_index(drop=True)

    user_item_mat = pd.DataFrame(0, index=freq_user, columns=freq_item)
    for i in range(len(data)):
        user_item_mat.loc[data['user'][i], data['movie'][i]] = data['rating'][i]

    return user_item_mat, freq_user, freq_item


def data_split(data, corrupt_ratio = 0.1):
    """
    corrupt data based on user, like if corrupt ratio is 0.1, then drop and save 1 out of 10 nonzero item rating
    :param data:
    :param corrupter_ratio:
    :return:
    """
    test_data = pd.DataFrame()
    for i in data.index:
        temp = data.loc[i,:]
        temp_nonzero = temp.nonzero()[0]
        select_index = list(np.random.choice(temp_nonzero, int(len(temp)*corrupt_ratio)))
        item_list = list(data.columns[select_index])

        corrupt_temp = [[i, item, data.loc[i,item]] for item in item_list]
        test_data = test_data.append(corrupt_temp)

        data.loc[i, item_list] = 0

    test_data.columns = ['user', 'item', 'rating']

    return data, test_data


def transfer_user_item_mat(data):
    """

    :param data:
    :return:
    """

    user_item_mat = pd.DataFrame(0, index=data.user, columns=data.movie)
    for i in range(len(data)):
        user_item_mat.loc[data['user'][i], data['movie'][i]] = data['rating'][i]

    return user_item_mat












