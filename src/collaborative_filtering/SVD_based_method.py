import numpy as np
from src.utilits.share import cosine_similarity
import pandas as pd
import warnings
from numpy.random import randint, normal

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
        return song_ranking.index[:num]
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


def cf_baseline(a, reg, step_size, max_iter, tol, check_point = None):
    """
    a baseline method for collborative filtering
    approx rating b_{ui} is given by b_{ui} = \mu + b_u + b_i
    true rating r_{ui} \aaprox b_{ui}
    try to get estimation of b_u and b_i by solving the least squares problem:
    min \sum (r_{ui} - \mu - b_u - b_i)^2 + \lambda_1 (\sum b_u^2 + \sum b_i^2)
    solve it by stochastic gradient
    b_u <- b_u + \gamma (r_{ui} - b_{ui} - \lambda_4 b_u)
    b_i <- b_i + \gamma (r_{ui} - b_{ui} - \lambda_4 b_i)
    :param a: the user-item matrix
    :param reg: parameter for regularization
    :param step_size: parameter for SGD
    :param max_iter: max epoch
    :return: b_u, b_i
    """
    # generate random variable
    random_index = [[i, j] for i in randint(a.shape[0], size=max_iter) for j in randint(a.shape[1], size=max_iter)]

    b_u = [0] * a.shape[0]
    b_i = [0] * a.shape[1]

    mu = np.mean(a)
    b = np.ones(a.shape) * mu

    error_mat = a - b

    for i in range(max_iter):
        m,n = random_index[i]
        b_u += step_size * (error_mat[:,n] - reg * b_u)
        b_i += step_size * (error_mat[m,:] - reg * b_i)

        b = np.ones(a.shape) * mu
        for x in range(len(b_u)):
            b[x, :] =  b_u[x]
        for y in range(len(b_i)):
            b[:, y] = b_i[y]

        error_mat = a - b

        loss = (np.sum(error_mat ** 2)/(a.shape[0] * a.shape[1]))**0.5
        if loss < tol:
            return b_u, b_i

    return b_u, b_i




def svd_sgd(a, k, reg, step_size, max_iter, tol):
    """
    try to minimize the regularized square error:
    min \sum(r_{ui} - \mu - b_i - b_u - q_i^Tp_u)^2 + \lambda_4(b_i^2 + b_u^2 + |q_i|^2 + |p_u|^2)
    :param a: the user-item matrix
    :param reg: parameter for regularization
    :param step_size: parameter for SGD
    :param max_iter: max epoch
    :param tol: the early stopping criterion
    :return: b_u, b_i, q p
    """

    # generate random variable
    random_index = [[i, j] for i in randint(a.shape[0], size=max_iter) for j in randint(a.shape[1], size=max_iter)]

    user_num = a.shape[0]
    item_num = a.shape[1]

    # generate initial q, p where each elemenet is normally distributed
    q = normal(0, 1, size=(item_num, k))
    p = normal(0, 1, size=(user_num, k))

    b_u = [0] * a.shape[0]
    b_i = [0] * a.shape[1]

    mu = np.mean(a)
    b = np.ones(a.shape) * mu +  p @ q.T

    for i in range(max_iter):
        m, n = random_index[i]
        b_u[m] += step_size * (error_mat[m, n] - reg * b_u[m])
        b_i[n] += step_size * (error_mat[m, n] - reg * b_i[n])
        q_old = q[n,:]
        q[n,:] += step_size * (error_mat[m, n]*p[m,:] - reg*q[n,:])
        p[m,:] += step_size * (error_mat[m, n]*q_old - reg*p[m,:])

        # maybe issue
        b[m, :] = [(mu + b_u[m])]*item_num + b_i + p @ q[n,:].reshape((k,1))
        b[:, n] = [(mu + b_i[n])]*user_num + b_u + q @ p[m,:].reshape((k,1))
        b[m, n] = mu + b_u[m] + b_i[n] + q[n,:] @ p[m,:]

        error_mat = a - b

        loss = (np.sum(error_mat ** 2) / (a.shape[0] * a.shape[1])) ** 0.5
        if loss < tol:
            return b_u, b_i

    return b_u, b_i, q, p



def svd_pp_sgd(a, k, reg, step_size, max_iter, tol):
    """
    try to minimize the regularized square error:
    min \sum(r_{ui} - \mu - b_i - b_u - q_i^Tp_u)^2 + \lambda_4(b_i^2 + b_u^2 + |q_i|^2 + |p_u|^2)
    :param a: the user-item matrix
    :param reg: parameter for regularization
    :param step_size: parameter for SGD
    :param max_iter: max epoch
    :param tol: the early stopping criterion
    :return: b_u, b_i, q p
    """

    # generate random variable
    random_index = [[i, j] for i in randint(a.shape[0], size=max_iter) for j in randint(a.shape[1], size=max_iter)]

    user_num = a.shape[0]
    item_num = a.shape[1]

    # generate initial q, p where each elemenet is normally distributed
    q = normal(0, 1, size=(item_num, k))
    p = normal(0, 1, size=(user_num, k))

    b_u = [0] * a.shape[0]
    b_i = [0] * a.shape[1]

    y_i = normal(0, 1, size=(item_num, k))

    mu = np.mean(a)
    b = np.ones(a.shape) * mu +  p @ q.T

    for i in range(max_iter):
        m, n = random_index[i]
        b_u[m] += step_size * (error_mat[m, n] - reg * b_u[m])
        b_i[n] += step_size * (error_mat[m, n] - reg * b_i[n])
        q_old = q[n,:]
        r_m = list(a[m,:])
        r_m = [(i>0)*1 for i in r_m ]
        y_i_m = np.array(r_m) @ y_i
        q[n,:] += step_size * (error_mat[m, n]*(p[m,:]+sum(r_m)**(-0.5)*np.sum(y_i_m, axis=0)) - reg*q[n,:])
        p[m,:] += step_size * (error_mat[m, n]*q_old - reg*p[m,:])
        for j in range(item_num):
            if r_m[j] != 0:
                y_i[j,:] += step_size * (error_mat[m, n]*sum(r_m)**(-0.5)*q[n,:] - reg*y_i[j,:])

        # in svd ++, have to update the whole matrix
        for mat_i in b.shape[0]:
            r_m = list(a[mat_i, :])
            r_m = [(i > 0) * 1 for i in r_m]
            y_i_m = np.array(r_m) @ y_i
            for mat_j in b.shape[1]:
                b[mat_i, mat_j] = mu + b_i[mat_j] + b_u[mat_i] \
                                  + q[mat_j,:] @ (p[mat_i,:] + r_m**(-0.5)*np.sum(y_i_m, axis=0))


        error_mat = a - b

        loss = (np.sum(error_mat ** 2) / (a.shape[0] * a.shape[1])) ** 0.5
        if loss < tol:
            return b_u, b_i

    return b_u, b_i, q, p


if __name__ == "__main__":
    from surprise import SVD
    from src.data.load_data import load_movielens_100k
    from src.collaborative_filtering.SVD_surprise import svd_predict
    from src.utilits.share import mse, mae
    from src.data.load_data import load_popular_sub_data

    data, data_train, data_test = load_movielens_100k()

    # surprise svd
    algo = SVD()
    algo.fit(data_train)
    test_pred = svd_predict(algo, data_test)
    print("RMSE:\t\t{}".format(mse(data_test, test_pred)),
          "MAE:\t\t{}".format(mae(data_test, test_pred)), sep='\n')

    # self code recommender
    core_data = load_popular_sub_data(data, 100, 100)



















