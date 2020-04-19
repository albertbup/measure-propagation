import numpy as np


def _euclidean_dist(p, q):
    """
    Returns euclidean distance given two numpy arrays.
    """
    return np.linalg.norm(p - q)


def euclidean_similarity(p, q):
    """
    Returns euclidean similarity given two numpy arrays.
    """
    return 1 / (1 + _euclidean_dist(p, q))


def cosine_similarity(p, q):
    """
    Returns cosine similarity given two numpy arrays.
    """
    return np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))
