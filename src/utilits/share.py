import numpy as np

def cosine_similarity(a, b):
    """
    calcualte the Cosine_simialrity between two vector
    :param a: array-like vecotr (n*1)
    :param b: array-like vector (n*1)
    :return:  a single value range from 0 to 1, represneting the similarity between a and b
    """
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))