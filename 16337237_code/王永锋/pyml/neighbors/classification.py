import numpy as np

from pyml.metrics.pairwise import euclidean_distance

class KNeighborsClassifier():
    """
    寻找 向量[x1, x2, x3, ... , xn] -> 种类 y（固定某范围） 的映射关系, y应该是非负整数

    Parameters
    ----------
    k : int, optional (default = 5)
        Number of neighbors to use by default.

    distance_func : callable, optional (default = euclidean_distance)
        distance function used in find the neighbors

        - [callable] : a function like f(vec1,vec2) 
            which can return the distance(float).

    Notes
    ------
    None

    """
    def __init__(self, k=5, distance_func=euclidean_distance):

        self.k = k
        self.distance_func = distance_func

    def fit(self, X, Y):
        """

        Parameters
        -----------
        X : array-like shape(n_samples, n_features)

        Y : array-like shape(n_samples,) not-negative number

        """
        self.X = X
        self.Y = Y
        self.n_features = self.X.shape[1] # the number of features in trained data
        self.n_samples = self.X.shape[0]

    def _predict_single(self, x):
        """

        Parameters
        -----------
        x : 1d array-like
            one sample

        Returns
        --------
        y_hat : the prediction of y

        """
        distances = np.zeros((self.n_samples))
        # calculate the distances between x and every sample in self.X
        for i, stored_x in enumerate(self.X):  # iterate through row vectors
            distances[i] = self.distance_func(stored_x, x)
        # print(distances)
        # find the indices of the k-smallest values through sort
        k_min_indexs = np.argsort(distances)[:self.k]
        k_min_distances = distances[k_min_indexs]
        # print(k_min_indexs)
        # print(k_min_distances)
        # 由距离最小的k个x对应的y
        k_min_ys = self.Y[k_min_indexs]
        # print(k_min_ys)

        # 保留顺序，寻找出现次数最多的一个，在出现次数相同的情况下使用距离较小的那个
        # 1. 先寻找出现次数最多的y的列表（可能会有重复）
        y_counts = np.bincount(k_min_ys)
        most_common_ys = np.argwhere(y_counts == np.amax(y_counts))
        most_common_ys = most_common_ys.reshape(-1)
        # print(most_common_ys)
        # 2. 在这些出现次数最多的y中，选择距离最小的y(即在k_min_ys里面排最前的那个)
        nearest_index = 999
        for most_common_y in most_common_ys:
            cur_index_tuple = np.where(k_min_ys == most_common_y)
            cur_index = cur_index_tuple[0][0]
            if cur_index < nearest_index:
                nearest_index = cur_index
        y_hat = k_min_ys[nearest_index]
        return y_hat


    def predict(self, X_pred):
        """

        Parameters
        -------------
            X_pred : 2d array-like shape(n_samples, n_feature)

        Returns
        --------------
            pre_Y : 1d array-like shape(n_samples,)

        """
        assert(X_pred.shape[1] == self.n_features) # the number of features should be consistent
        Y_pred = np.zeros((X_pred.shape[0]))
        for i,x_pred in enumerate(X_pred):
            Y_pred[i] = self._predict_single(x_pred)
        return Y_pred
