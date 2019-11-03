import numpy as np
from src.utilits.share import cosine_similarity
import pandas as pd

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

def N_top_recommender_svd(a, neigbor, num, if_user = True, method = 'freq'):
    """
    return the recommend item
    :param a: the original user by item matrix
    :param neigbor: list, the neigbor index
    :param num: how many
    :param if_user: if the neigbor refers to user, then true. if item, then false
    :param method: param only works when if_user = True, two options: 'freq' refers to count the top
    :return:
    """








