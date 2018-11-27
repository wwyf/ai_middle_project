import numpy as np

class StandardScaler():
    """
    将数据归一化
    """
    def __init__(self):
        return
    def fit(self, X : np.ndarray):
        """
        Parameters
        -----------
        X : 2d array-like shape(n_samples, n_features)

        """
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return
    
    def transform(self, X, mean=0, std=1):
        normal =  (X - X.mean(axis=0))/(X.std(axis=0)+0.00000001)
        return std * (normal + mean)

    def fit_transform(self, X, mean=0, std=1):
        self.fit(X)
        return self.transform(X, mean, std)

def scale(X):
    """
    Standardize a dataset along any axis
    Center to the mean and component wise scale to unit variance.

    """
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    return (X - X.mean(axis=0))/(X.std(axis=0)+0.0000001)
    

def MinMaxScaler(X):
    """
    (X-X.min())/(X.max()-X.min())
    
    Parameters
    -----------
    X : 2d array-like shape(n_samples, n_features)

    Returns
    ----------
    X : 2d array-like shape(n_samples, n_feature)
        values will be in [0,1]
    """
    return (X - X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))

def normalize():
    pass

if __name__ == '__main__':
     a = np.array([[0,1,5],[3,9,1],[4,5,6]])
     ss = StandardScaler()
     print(ss.fit_transform(a))
"""
[[-1.37281295 -1.22474487  0.46291005]
 [ 0.39223227  1.22474487 -1.38873015]
 [ 0.98058068  0.          0.9258201 ]]
"""
