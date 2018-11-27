import numpy as np


def cosine_similarity(vec1 : np.ndarray, vec2 : np.ndarray):
    """the cosine similarity of two vectors

    using this formula : $$ cosine_similarity(x,y) = \frac{ |x·y| } { |x|·|y| } $$

    Parameters
    ----------

    vec1: shape (1, n_features) or shape(n_features,)

    vec2: shape (1, n_features) or shape(n_features,)

    Returns
    --------

    distances : shape (1, 1)

    """
    vec1 = vec1.reshape((1,-1))
    vec2 = vec2.reshape((1,-1))
    assert(vec1.shape[0] == 1)
    assert(vec2.shape[0] == 1)
    assert(vec1.shape[1] == vec2.shape[1])
    mult_sum = (vec1 * vec2).sum()
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    eps = 0.000000001
    # print(mult_sum)
    return mult_sum/(norm_vec1 * norm_vec2 + eps)

def cosine_distance(vec1, vec2):
    """the cosine distance of two vectors

    Cosine distance is defined as 1.0 minus the cosine similarity.

    Parameters
    ----------

    vec1 : shape (1, n_features) or shape(n_features,)

    vec2 : shape (1, n_features) or shape(n_features,)

    Returns
    --------

    distances : shape (1, 1)

    """
    return 1.0 - cosine_similarity(vec1, vec2)

def euclidean_distance(vec1, vec2):
    """the euclidean distance of two vectors

    Parameters
    -----------

    vec1: shape (1, n_features) or shape(n_features,)

    vec2: shape (1, n_features) or shape(n_features,)

    Returns
    -------

    distances : shape (1, 1)

    """

    vec1 = vec1.reshape((1,-1))
    vec2 = vec2.reshape((1,-1))
    assert(vec1.shape[0] == 1)
    assert(vec2.shape[0] == 1)
    assert(vec1.shape[1] == vec2.shape[1])
    return np.linalg.norm(vec1-vec2)

def absolute_distance(vec1, vec2):
    """the euclidean distance of two vectors

    Parameters
    -----------

    vec1: shape (1, n_features) or shape(n_features,)

    vec2: shape (1, n_features) or shape(n_features,)

    Returns
    -------

    distances : shape (1, 1)

    """
    vec1 = vec1.reshape((1,-1))
    vec2 = vec2.reshape((1,-1))
    assert(vec1.shape[0] == 1)
    assert(vec2.shape[0] == 1)
    assert(vec1.shape[1] == vec2.shape[1])
    return np.abs(vec1-vec2).sum()

def l_p_distance(vec1, vec2,p=1):
    """ calculate the p(default 1) norm of two vectors

    Parameters
    ------------

    vec1 : shape (1, n_features) or shape(n_features,)

    vec2 : shape (1, n_features) or shape(n_features,)

    Returns
    --------

    distances : shape (1, 1)

    """

    vec1 = vec1.reshape((1,-1))
    vec2 = vec2.reshape((1,-1))
    assert(vec1.shape[0] == 1)
    assert(vec2.shape[0] == 1)
    assert(vec1.shape[1] == vec2.shape[1])
    if (p == np.Inf):
        return np.max(np.abs(vec1-vec2))
    return np.power(np.power(np.abs(vec1-vec2),p).sum(),1/p)