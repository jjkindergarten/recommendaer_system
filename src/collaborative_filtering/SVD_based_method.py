import numpy as np
from src.utilits.share import cosine_similarity
import pandas as pd
import warnings
from random import shuffle
from numpy.random import randint, normal, choice
from src.utilits.share import generate_error_matrix

def _svd(a, preservation):
    """
    apply svd on n by m matrix return user column space n by k and item column space m by k
    :param a: n by m matrix
    :param preservation: the percentage or number of singular value needed to be preserved
    :return: user column space n by k and item column space m by k and singular matrix k by k
    """

    u, s, vh = np.linalg.svd(a, full_matrices=False)
    v = vh.T

    if preservation < 1:
        i = 0
        singular_value_sum = 0
        while singular_value_sum < preservation*sum(s):
            singular_value_sum += s[i]
            i += 1
    else:
        i = int(preservation)

    return u[:,:i], v[:,:i], np.diag(s[:i])

def k_neighbor_svd(target_index, k, char_m, s_matrix):
    """
    find the k cloest neigbor of a specific user or item using cosine similarity
    :param target_index: a number smaller than n (user) or m (item)
    :param k: number of neigbor
    :param char_m: the characteristic matrix
    :param s_matrix: the characteristic matrix
    :return: a list contains all user\itme index
    """

    if k >= len(char_m)-1:
        return [i for i in range(len(char_m)) if i != target_index]

    char_m = char_m @ s_matrix
    similarity_list = []
    target = char_m[target_index,:]
    for i in range(len(char_m)):
        if i != target_index:
            similarity = cosine_similarity(target, char_m[i,:])
            similarity_list.append([i, similarity])

    similarity_db = pd.DataFrame(similarity_list, columns=['index', 'similarity'])
    similarity_db = similarity_db.sort_values('similarity', ascending = False)
    temp = similarity_db.iloc[:k, :]

    return list(temp['index'])

def N_top_recommender_svd(a, neigbor, num, if_user = True):
    """
    return the recommend item
    :param a: the original user by item matrix
    :param neigbor: list, the neigbor index
    :param num: how many
    :param if_user: if the neigbor refers to user, then true. if item, then false
    :param method: param only works when if_user = True, two options: 'freq' refers to count the top
    :return: return the index of name of recommendation
    """


    if if_user:
        neigbor_user = a.iloc[neigbor,:]
        song_ranking = neigbor_user.mean(axis = 0)
        song_ranking.sort_values(ascending=False, inplace = True)
        return list(song_ranking.index[:num])
    else:
        # For item, it would return the cloest neigbor directly.
        # The k-cloest neigbor should be consistent with N-recommendation in this case.
        return list(a.columns)[neigbor]

def insert_new_user(new_user, user_char_mat, s):
    """
    insert new user into the user characteristic matrix
    :param new_user: 1 by m vecotr, where m denotes the number of item
    :param user_char_mat: the original user characteristic matrix (m by k)
    :param s: k by k diagonal matrix
    :return: new user characteristic matrix with new user vector inserted at the end
    """

    new_user_v = new_user @ user_char_mat @ np.linalg.inv(s)
    return np.vstack((user_char_mat, new_user_v))

def cf_baseline(a, reg, step_size, max_iter, tol):
    """
    a baseline method for collborative filtering
    approx rating b_{ui} is given by b_{ui} = \mu + b_u + b_i
    true rating r_{ui} \aaprox b_{ui}
    try to get estimation of b_u and b_i by solving the least squares problem:
    min \sum (r_{ui} - \mu - b_u - b_i)^2 + \lambda_1 (\sum b_u^2 + \sum b_i^2)
    solve it by stochastic gradient
    b_u <- b_u + \gamma (r_{ui} - b_{ui} - \lambda_4 b_u)
    b_i <- b_i + \gamma (r_{ui} - b_{ui} - \lambda_4 b_i)
    :param a: the user-item matrix (compressed one)
    :param reg: parameter for regularization
    :param step_size: parameter for SGD
    :param max_iter: max epoch
    :return: b_u, b_i
    """
    # generate random variable
    user_list = list(a.index)
    item_list = list(a.columns)
    a = a.values
    index_u, index_i = a.nonzero()
    nonzero_index = [[index_u[i], index_i[i]] for i in range(len(index_u))]
    random_index = [nonzero_index[i] for i in choice(range(len(nonzero_index)), max_iter, replace = True)]

    b_u = np.zeros(a.shape[0])
    b_i = np.zeros(a.shape[1])

    mu = np.sum(a)/len(np.nonzero(a)[0])
    b = np.ones(a.shape) * mu

    for i in range(max_iter):
        m, n = random_index[i]
        error = a[m,n] - b[m,n]
        pre_b_u_m = b_u[m]
        pre_b_i_n = b_i[n]
        b_u[m] = b_u[m] + step_size * (error - reg * b_u[m])
        b_i[n] = b_i[n] + step_size * (error - reg * b_i[n])

        b[m, :] = b[m, :] - pre_b_u_m + b_u[m]
        b[:, n] = b[:, n] - pre_b_i_n + b_i[n]

        if i % 100 == 0:
            error_mat = generate_error_matrix(a, b)
            loss = np.linalg.norm(error_mat,2)/(len(error_mat) ** 0.5)
            print('loss is {}'.format(loss))

    b_u = pd.DataFrame(b_u, index = user_list, columns=['user_sd'])
    b_i = pd.DataFrame(b_i, index = item_list, columns=['item_sd'])

    return mu, b_u, b_i, b

def svd_sgd(a, k, reg, step_size, max_iter, tol):
    """
    try to minimize the regularized square error:
    min \sum(r_{ui} - \mu - b_i - b_u - q_i^Tp_u)^2 + \lambda_4(b_i^2 + b_u^2 + |q_i|^2 + |p_u|^2)
    :param a: the user-item matrix
    :param k: characteristic matrix's feature num
    :param reg: parameter for regularization
    :param step_size: parameter for SGD
    :param max_iter: max epoch
    :param tol: the early stopping criterion
    :return: b_u, b_i, q p
    """

    # generate random variable
    user_list = list(a.index)
    item_list = list(a.columns)
    a = a.values
    index_u, index_i = a.nonzero()
    nonzero_index = [[index_u[i], index_i[i]] for i in range(len(index_u))]
    random_index = [nonzero_index[i] for i in choice(range(len(nonzero_index)), max_iter, replace = True)]

    user_num = a.shape[0]
    item_num = a.shape[1]

    # generate initial q, p where each elemenet is normally distributed
    q = normal(0, 1, size=(item_num, k))
    p = normal(0, 1, size=(user_num, k))

    b_u = np.zeros(a.shape[0])
    b_i = np.zeros(a.shape[1])

    mu = np.sum(a)/len(np.nonzero(a)[0])
    b = np.ones(a.shape) * mu +  p @ q.T

    for i in range(max_iter):
        m, n = random_index[i]
        error = a[m,n] - b[m,n]

        pre_b_u_m = b_u[m]
        pre_b_i_n = b_i[n]
        pre_q = q.copy()
        pre_p = p.copy()

        b_u[m] += step_size * (error - reg * b_u[m])
        b_i[n] += step_size * (error - reg * b_i[n])
        q[n,:] = q[n,:] + step_size * (error*p[m,:] - reg*q[n,:])
        p[m,:] = p[m,:] + step_size * (error*pre_q[n,:] - reg*p[m,:])

        # maybe issue
        b[m, :] += -pre_b_u_m + b_u[m]
        b[:, n] += -pre_b_i_n + b_i[n]
        b[:, n] += -pre_p @ pre_q[n,:] + p @ q[n,:]
        b[m, :] += -pre_q @ pre_p[m,:] + q @ p[m,:]

        if i % 100 == 0:
            error_mat = generate_error_matrix(a, b)
            loss = np.linalg.norm(error_mat,2)/(len(error_mat) ** 0.5)
            print('loss is {}'.format(loss))

    b_u = pd.DataFrame(b_u, index = user_list, columns=['user_sd'])
    b_i = pd.DataFrame(b_i, index = item_list, columns=['item_sd'])
    q = pd.DataFrame(q, index = item_list)
    p = pd.DataFrame(p, index = user_list)

    return mu, b_u, b_i, q, p


def baseline_predict(data, mu, b_u, b_i):
    """

    :param data:
    :param mu:
    :param b_u:
    :param b_i:
    :return:
    """
    user_list = list(b_u.index)
    item_list = list(b_i.index)
    data = data[data.user.isin(user_list)]
    data = data[data.movie.isin(item_list)]

    data_pred = data.copy()
    data_pred['rating'] = 0
    for i in range(len(data)):
        user = data_pred['user'][i]
        item = data_pred['movie'][i]

        data_pred.loc[i, 'rating'] = mu + b_u.loc[user, 'user_sd'] + b_i.loc[item, 'item_sd']
    return data_pred


def svd_predict(data, mu, b_u, b_i, q, p):
    """

    :param data:
    :param mu:
    :param b_u:
    :param b_i:
    :return:
    """
    user_list = list(b_u.index)
    item_list = list(b_i.index)
    data = data[data.user.isin(user_list)]
    data = data[data.movie.isin(item_list)]

    data_pred = data.copy()
    data_pred['rating'] = 0
    for i in range(len(data)):
        user = data_pred['user'][i]
        item = data_pred['movie'][i]

        data_pred.loc[i, 'rating'] = mu + b_u.loc[user, 'user_sd'] + b_i.loc[item, 'item_sd'] + p.loc[user,:] @ q.loc[item,:]
    return data_pred



if __name__ == "__main__":
    from surprise import SVD, BaselineOnly, Dataset, Reader
    from src.data.load_data import load_movielens_100k, transfer_user_item_mat
    from src.collaborative_filtering.SVD_surprise import surprise_predict
    from src.utilits.share import mse, mae
    from src.data.load_data import load_popular_sub_data, data_split

    data, _, _ = load_movielens_100k()

    core_data, train_data_ui, data_train, data_test = load_popular_sub_data(data, 100, 200, 0.75)
    data_train_sp = Dataset.load_from_df(data_train, reader=Reader('ml-100k')).build_full_trainset()

    ################# self code recommender ########################
    test_data = data[data.user.isin(list(core_data.index))]
    test_item = [i for i in data.movie if i not in list(core_data.columns)]
    test_data = test_data[test_data.movie.isin(test_item)]
    _, test_data, _, _ = load_popular_sub_data(test_data, 100, 100)
    user_mat, item_mat, singular_mat = _svd(train_data_ui, 10)

    # check user similarity
    close_neibor_index = k_neighbor_svd(1, 10, user_mat, singular_mat)
    close_neibor = list(train_data_ui.index[close_neibor_index])
    print('close niegbor of {} is: {}'.format(core_data.index[1], close_neibor))

    recom_item = N_top_recommender_svd(test_data, close_neibor_index, 10, if_user=True)
    user = core_data.index[1]
    data.loc[(data['user'] == user) & (data['movie'].isin(recom_item)), :]

    ############## self code baseline ##################
    # try surprise baseline first
    base_algo = BaselineOnly()
    base_algo.fit(data_train_sp)
    test_pred_base = surprise_predict(base_algo, data_test)
    print("RMSE:\t\t{}".format(mse(data_test, test_pred_base)),
          "MAE:\t\t{}".format(mae(data_test, test_pred_base)), sep='\n')

    # self code baseline
    mu, b_u, b_i, b = cf_baseline(train_data_ui, reg=0.004, step_size = 0.01, max_iter = 50000, tol = 1e-4)
    data_test_base_pred = baseline_predict(data_test, mu, b_u, b_i)
    print("RMSE:\t\t{}".format(mse(data_test, data_test_base_pred)),
          "MAE:\t\t{}".format(mae(data_test, data_test_base_pred)), sep='\n')


    ############# self code svd ###################
    # surprise svd
    algo = SVD()
    algo.fit(data_train_sp)
    test_pred_SVD = surprise_predict(algo, data_test)
    print("RMSE:\t\t{}".format(mse(data_test, test_pred_SVD)),
          "MAE:\t\t{}".format(mae(data_test, test_pred_SVD)), sep='\n')

    mu, b_u, b_i, q, p = svd_sgd(train_data_ui, k=5, reg=0.05, step_size=0.02, max_iter=200000, tol=1e-4)
    data_test_svd_pred = svd_predict(data_test, mu, b_u, b_i, q, p)
    print("RMSE:\t\t{}".format(mse(data_test, data_test_svd_pred)),
          "MAE:\t\t{}".format(mae(data_test, data_test_svd_pred)), sep='\n')




















