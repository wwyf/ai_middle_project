from pyml.tree.regression import DecisionTreeRegressor
from pyml.metrics.pairwise import euclidean_distance
import numpy as np
# TODO: 使用平方误差，还是绝对值误差，还是Huber Loss

class GradientBoostingRegression():
    def __init__(self,
        learning_rate=0.1,
        base_estimator=DecisionTreeRegressor,
        n_estimators=500,
        random_state=None
    ):
        self.estimators = []
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.learning_rate = learning_rate
        self.parameters = {
            'f' : [],
            'lr' : []
        }
        # key='f' : a list of estimator
        # key='lr' : a list of learning_rate

    def optimizer(self, X, Y, watch=False):
        """
        训练一次
        """
        cur_Y_pred = self.predict(X)
        # print('cur_Y_pred : ', cur_Y_pred)

        # 计算cost
        cost = euclidean_distance(cur_Y_pred, Y)

        # 计算残差 or 计算梯度
        d_fx = cur_Y_pred - Y
        # print('d_fx : ', d_fx)

        # 梯度取负数
        d_fx = - d_fx

        # 计算学习率，这里默认为初始化参数
        lr = self.learning_rate

        # 创建一个新回归器，去拟合梯度
        new_estimator = self.base_estimator()
        new_estimator.fit(X,d_fx)
        self.parameters['f'].append(new_estimator)
        self.parameters['lr'].append(lr)
        return cost

    def fit(self, X, Y, watch=False):
        init_estimator = self.base_estimator()
        init_estimator.fit(X,Y)
        self.parameters['f'].append(init_estimator)
        self.parameters['lr'].append(1)
        for i in range(self.n_estimators):
            cost = self.optimizer(X,Y)
            if i % 10 == 0:
                print('train {}/{}  current cost : {}'.format(i,self.n_estimators,cost))

    def predict(self, X_pred):
        """

        Parameters
        -------------
            X_pred : 2d array-like shape(n_samples, n_feature)

        Returns
        --------------
            pre_Y : 1d array-like shape(n_samples,)

        """
        # the number of features should be consistent.
        total_num = X_pred.shape[0]
        Y_pred = np.zeros((total_num))
        for cur_estimator, lr in zip(self.parameters['f'], self.parameters['lr']):
            Y_pred += cur_estimator.predict(X_pred) * lr
        return Y_pred

if __name__ == '__main__':
    mini_train_X = np.array([
        [1,2,3,4,5,6,7,8],
        [2,3,4,5,6,7,8,9],
        [3,4,5,6,7,8,9,10],
        [4,5,6,7,8,9,10,11],
        [5,6,7,8,9,10,11,12],
        [6,7,8,9,10,11,12,13],
        [7,8,9,10,11,12,13,14]
    ])
    mini_train_Y = np.array([
        1.5,2.5,3.5,4.5,5.5,6.5,7.5
    ])
    mini_test_X = np.array([
        [2,3,4,5,6,7.5,8,9],
        [4,5,6,7.5,8,9,10,11]
    ])
    mini_standard_out_Y = np.array([
        2.5,4.5
    ])
    rgs = GradientBoostingRegression()
    rgs.fit(mini_train_X,mini_train_Y)
    print(rgs.predict(mini_test_X))