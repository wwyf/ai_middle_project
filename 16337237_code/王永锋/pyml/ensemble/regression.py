from pyml.tree.regression import DecisionTreeRegressor
from pyml.metrics.pairwise import euclidean_distance
from pyml.metrics.pairwise import absolute_distance
from pyml.metrics.regression import pearson_correlation
import numpy as np
from pyml.logger import logger
import math

# TODO: 使用平方误差，还是绝对值误差，还是Huber Loss
# TODO: 如何给基回归器传参

def random_mini_batches(X, Y, mini_batch_size = 64, seed = None):
    """
    X : shape(n_samples,n_features)
    Y : shape(n_samples,)

    batch:
    shape(n_samples, n_features) shape(n_samples)
    """
    if seed is not None:
        np.random.seed(seed)
    X = X.T
    m = X.shape[1]
    mini_batches = []
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[permutation].reshape(-1)
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X.T, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X.T, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches



class GradientBoostingRegression():
    def __init__(self,
        loss='ls',
        learning_rate=0.1,
        base_estimator=DecisionTreeRegressor,
        n_estimators=100,
        max_tree_node_size=10,
        delta=0.5,
        random_state=None,
        base_divide_way='default',
        divide_way='half',
        mini_batch=64,
        print_interval=10
    ):
        """
        Parameters
        ------------
        loss : loss function to be optimized
            - 'ls' least squares
            - 'lad' least absolute deviation
            - 'huber' ???
        """
        self.loss = loss
        self.estimators = []
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.learning_rate = learning_rate
        self.max_tree_node_size = max_tree_node_size
        self.delta=delta
        self.base_divide_way = base_divide_way
        self.divide_way = divide_way
        self.mini_batches = []
        self.current_mini_batch_index = 0
        self.mini_batch = mini_batch
        self.print_interval = print_interval
        # key='f' : a list of estimator
        # key='lr' : a list of learning_rate
        self.information = {
            'gradient':[],
            'valid_loss':[],
            'train_loss':[]
        }
        self.parameters = {
            'f' : [],
            'lr' : []
        }

    def init_mini_batches(self, X, Y):
        """
        X : 2d array-like shape(n_samples,n_features)
        Y : 2d array-like shape(n_samples,1)
        """
        self.mini_batches = random_mini_batches(X, Y, self.mini_batch)
        self.current_mini_batch_index = 0

    def get_mini_batch(self):
        current_mini_batch = self.mini_batches[self.current_mini_batch_index]
        return current_mini_batch
    def update_mini_batch(self):
        self.current_mini_batch_index = (self.current_mini_batch_index+1) % len(self.mini_batches)

    def optimizer(self, X, Y, watch=False):
        """
        训练一次
        """
        logger.debug('X : \n{}\nY : {}'.format(X, Y))
        cur_Y_pred = self.predict(X)
        logger.debug('cur_Y_pred : {}'.format(cur_Y_pred))

        if self.loss == 'ls':
            # 计算均方误差，平方和除2（除2是为了与之后的梯度对应）
            cost = np.square(cur_Y_pred- Y).sum()/2
            # 计算残差 or 计算梯度
            d_fx = cur_Y_pred - Y
            logger.debug('d_fx : {}'.format(d_fx))
            # 梯度取负数
            d_fx = - d_fx
        elif self.loss == 'lad':
            cost = absolute_distance(cur_Y_pred, Y)
            d_fx = np.sign(cur_Y_pred-Y)
            d_fx = - d_fx
        elif self.loss == 'huber':
            # 计算cost
            deviation = cur_Y_pred-Y
            logger.debug('deviation : {}'.format(deviation))
            abs_deviation = np.abs(deviation)
            logger.debug('abs_deviation : {}'.format(abs_deviation))
            small_part_index = abs_deviation <= self.delta
            big_part_index = abs_deviation > self.delta
            # 取得差小于等于delta的部分
            cost = np.square(abs_deviation[small_part_index]).sum()
            logger.debug('cost : {}'.format(cost))
            # 取得差大于delta的部分
            cost += self.delta*(abs_deviation[big_part_index]- self.delta/2).sum()
            logger.debug('cost : {}'.format(cost))
            d_fx = np.zeros((Y.shape))
            d_fx[small_part_index] = deviation[small_part_index]
            logger.debug('d_fx : {}'.format(d_fx))
            d_fx[big_part_index] = self.delta * np.sign(deviation[big_part_index])
            logger.debug('d_fx : {}'.format(d_fx))
            d_fx = -d_fx
        else:
            raise NotImplementedError

        # 计算学习率，这里默认为初始化参数
        lr = self.learning_rate
        self.information['gradient'].append(d_fx)

        # 创建一个新回归器，去拟合梯度
        new_estimator = self.base_estimator(
            max_node_size=self.max_tree_node_size,
            divide_way=self.divide_way
        )
        new_estimator.fit(X,d_fx)
        self.parameters['f'].append(new_estimator)
        self.parameters['lr'].append(lr)
        return cost

    def fit(self, X, Y,watch=False):
        logger.debug('X : \n{} Y : {}'.format(X, Y))
        init_estimator = self.base_estimator(
            max_node_size=self.max_tree_node_size,
            divide_way=self.base_divide_way
            )
        init_estimator.fit(X,Y)
        self.parameters['f'].append(init_estimator)
        self.parameters['lr'].append(1)
        for i in range(self.n_estimators):
            cost = self.optimizer(X,Y)
            if i % self.print_interval == 0:
                logger.info('train {}/{}  current cost: {}'.format(i,self.n_estimators,cost))
                # print('train {}/{}  current cost : {}'.format(i,self.n_estimators,cost))

    def fit_and_valid(self, X, Y,X_valid,Y_valid, watch=False):
        self.init_mini_batches(X,Y)
        logger.debug('X : \n{} Y : {}'.format(X, Y))
        init_estimator = self.base_estimator(
            max_node_size=self.max_tree_node_size,
            divide_way=self.base_divide_way
        )
        init_estimator.fit(X,Y)
        self.parameters['f'].append(init_estimator)
        self.parameters['lr'].append(1)
        for i in range(self.n_estimators):
            if self.mini_batch==0:
                # 使用全部样例：
                cost = self.optimizer(X,Y)
            else :
                # 使用mini_batch个样例：
                X_batch,Y_batch = self.get_mini_batch()
                cost = self.optimizer(X_batch, Y_batch)
            if i % self.print_interval == 0:
                this_loss =  self.get_test_cost(X_valid, Y_valid)
                train_loss =  self.get_test_cost(X, Y)
                self.information['valid_loss'].append(this_loss)
                self.information['train_loss'].append(train_loss)
                logger.info('train {}/{}  current cost: {},train : {} valid : {}'.format(i,self.n_estimators,cost,train_loss,this_loss))
                # print('train {}/{}  current cost : {}'.format(i,self.n_estimators,cost))
            self.update_mini_batch()

    def get_test_cost(self, X, Y):
        Y_pred = self.predict(X)
        return pearson_correlation(Y_pred, Y)

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