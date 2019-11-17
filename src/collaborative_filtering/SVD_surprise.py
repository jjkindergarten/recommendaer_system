import surprise
import pandas as pd

def svd_predict(alg, data):
    """

    :param alg: already trainning model
    :param data: test data
    :return: the prediction
    """
    pred_list = []
    for user_id in data['user'].unique():
        for movie_id in data['movie'].unique():
            pred_list.append([user_id, movie_id, alg.predict(user_id, movie_id).est])

    return pd.DataFrame(pred_list, columns=['user', 'movie', 'rating'])