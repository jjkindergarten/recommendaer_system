import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error


def cosine_similarity(a, b):
    """
    calcualte the Cosine_simialrity between two vector
    :param a: array-like vecotr (n*1)
    :param b: array-like vector (n*1)
    :return:  a single value range from 0 to 1, represneting the similarity between a and b
    """
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def mse(data, pred):
    """

    :param data:
    :param pred:
    :return:
    """
    suffixes = ["_true", "_pred"]
    rating_true_pred = data.merge(pred, on = ['user', 'movie'], how='inner', suffixes = suffixes)

    return np.sqrt(mean_squared_error(rating_true_pred['rating' + suffixes[0]], rating_true_pred['rating' + suffixes[1]]))

def mae(data, pred):
    """

    :param data:
    :param pred:
    :return:
    """
    suffixes = ["_true", "_pred"]
    rating_true_pred = data.merge(pred, on = ['user', 'movie'], how='inner', suffixes = suffixes)

    return np.sqrt(mean_absolute_error(rating_true_pred['rating' + suffixes[0]], rating_true_pred['rating' + suffixes[1]]))

def generate_error_matrix(a, b):
    """

    :param a:
    :param b:
    :return:
    """
    entry_diff = []
    row_index_set, col_index_set = a_nonzero_index = np.nonzero(a)
    for i in range(len(row_index_set)):
        row_index = row_index_set[i]
        col_index = col_index_set[i]

        entry_diff.append(a[row_index, col_index] - b[row_index, col_index])

    return np.array(entry_diff)
