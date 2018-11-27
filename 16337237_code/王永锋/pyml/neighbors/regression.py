import numpy as np

from pyml.metrics.pairwise import euclidean_distance

class KNeighborsRegressor():
    """
    寻找 向量[x1, x2, x3, ... , xn] -> 向量[y1, y2, y3, ... ym]的映射关系

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

        Y : array-like shape(n_samples, y_features)

        """
        self.X = X
        self.Y = Y
        self.n_samples = self.X.shape[0] # the number of samples
        self.n_features = self.X.shape[1] # the number of features in trained data
        self.y_features = self.Y.shape[1] # the number of features in label

    def _predict_single(self, x):
        """
        input :
            x : an numpy array
        return :
            y_hat : the prediction of y
        """
        distances = np.zeros((self.n_samples))
        # calculate the distances between x and every sample in self.X
        for i,stored_x in enumerate(self.X):
            distances[i] = self.distance_func(stored_x, x)
        # print(distances)

        # find the indices of the k-smallest values through sort
        k_min_indexs = np.argsort(distances)[:self.k]
        k_min_distances = distances[k_min_indexs]
        # print(k_min_indexs)
        # 由距离最小的k个x对应的y
        k_min_ys = self.Y[k_min_indexs]
        # print("k_min_ys\n", k_min_ys)
        k_min_distances = distances[k_min_indexs]
        k_min_distances = np.array(k_min_distances).reshape(-1,1)
        # 对距离先做正则化，然后再取倒数，然后再做正则化
        # 结果需要：1. 和为1 2. 距离与结果成反相关
        # TODO: 修改了这一处，还需要修改单元测试
        k_min_distances_normalized = (k_min_distances-np.amin(k_min_distances) + 0.1)/(np.amax(k_min_distances)-np.amin(k_min_distances) + 0.1)
        temp = 1/k_min_distances_normalized
        k_min_distances_normalized = temp/temp.sum()
        # 计算概率并归一化
        final_result = (k_min_ys * k_min_distances_normalized).sum(axis=0)
        # print(k_min_ys/(k_min_distances+1))
        # print("k_min_distances\n",k_min_distances)
        # print(k_min_ys)
        # print(final_result)
        # 这一行误事！
        # final_result = final_result/final_result.sum()
        return final_result.reshape(1,-1)
    def predict(self, X_pred, watch=False):
        """

        Parameters
        -------------
            X_pred : 2d array-like shape(n_samples, n_feature)

        Returns
        --------------
            pre_Y : 1d array-like shape(n_samples,)

        """
        assert(X_pred.shape[1] == self.n_features) # the number of features should be consistent
        Y_pred = np.zeros((X_pred.shape[0], self.y_features))
        total_num = X_pred.shape[0]
        count = 0
        for i,x_pred in enumerate(X_pred):
            Y_pred[i] = self._predict_single(x_pred)
            # print(Y_pred[i])
            count += 1
            if (watch and count % 10 == 0):
                print("current status: {}/{}".format(count, total_num))
        return Y_pred
