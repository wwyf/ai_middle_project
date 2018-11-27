import numpy as np
from pyml.metrics.classification import precision_score
from pyml.logger import logger
import math

def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters
    -------------
    z : A scalar or numpy array of any size.

    Returns
    ------------
    s : sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s


def random_mini_batches(X, Y, mini_batch_size = 64, seed = None):
    """
    X : shape(n_features, n_samples)
    Y : shape(1, n_samples)
    """
    if seed is not None:
        np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

class LogisticClassifier():
    """
    二元分类器，分类结果为0,1
    """
    def  __init__(self, learning_rate=0.1, num_iterations=2000, mini_batch=0, lambda_l2=0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.mini_batch = mini_batch
        self.lambda_l2 = lambda_l2
        self.parameters = {}
        self.information = {
            'test_loss' : [],
            'train_loss' : [],
            'cost' : []
        }
        self.mini_batches = []
        self.current_mini_batch_index = 0
    
    def initialize_parameters(self, n_features):
        """
        initialize the parameters with zeros

        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)

        """
        parameters = {}
        # TODO: 参数如何初始化
        parameters['w'] = np.random.normal(size=(n_features, 1))
        parameters['b'] = np.random.normal()
        # parameters['w'] = np.zeros((n_features, 1))
        # parameters['b'] = 0
        return parameters

    def init_mini_batches(self, X, Y):
        """
        X : 2d array-like shape(n_features, n_samples)
        Y : 2d array-like shape(1, n_samples)
        """
        self.mini_batches = random_mini_batches(X, Y, self.mini_batch)
        self.current_mini_batch_index = 0

    def get_mini_batch(self):
        current_mini_batch = self.mini_batches[self.current_mini_batch_index]
        self.current_mini_batch_index = (self.current_mini_batch_index+1) % len(self.mini_batches)
        return current_mini_batch

    def propagate(self, w, b, X, Y):
        """
        Parameters
        ----------
        w : 2d array-like shape(n_features, 1)
        b : 1d array-like shape(1,)
        X : 2d array-like shape(n_features, n_samples)
        Y : 1d array-like shape(1, n_samples)

        Returns
        -------
        dw : same shape as w
        """
        m = X.shape[1]
        A = sigmoid(np.dot(w.T, X) + b)
        l2 = self.lambda_l2 * (np.square(w).sum()+np.square(b).sum())/m
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) + l2
        dw = 1 / m * np.dot(X, (A - Y).T) + self.lambda_l2 * w/m
        db = 1 / m * np.sum(A - Y) + self.lambda_l2 * b/m
        cost = np.squeeze(cost)
        grads = {"dw": dw,
                "db": db}
        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations, print_cost=False):
        """
        Parameters
        ----------
        w : 2d array-like shape(n_features, 1)
        b : 1d array-like shape(1,)
        X : 2d array-like shape(n_features, n_samples)
        Y : 1d array-like shape(1, n_samples)
        """
        learning_rate = self.learning_rate
        costs = []
        for i in range(num_iterations):
            grads, cost = self.propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]
            w = w - learning_rate * dw
            b = b - learning_rate * db
            if i % 100 == 0:
                costs.append(cost)
            # Print the cost every 100 training examples
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
        params = {"w": w,
                "b": b}
        grads = {"dw": dw,
                "db": db}
        return params, grads, costs

    def optimize_single(self, w, b, X, Y):
        """
        Parameters
        ----------
        w : 2d array-like shape(n_features, 1)
        b : 1d array-like shape(1,)
        X : 2d array-like shape(n_features, n_samples)
        Y : 1d array-like shape(1, n_samples)
        """
        logger.debug('X : {}\nshape:{}'.format(X,X.shape))
        logger.debug('Y : {}\nshape:{}'.format(Y,Y.shape))
        learning_rate = self.learning_rate
        mini_batch = self.mini_batch
        if self.mini_batch == 0:
            grads, cost = self.propagate(w, b, X, Y)
        else:
            X_batch,Y_batch = self.get_mini_batch()
            logger.debug('X_batch : {}\nshape:{}'.format(X_batch,X_batch.shape))
            logger.debug('Y_batch : {}\nshape:{}'.format(Y_batch,Y_batch.shape))
            grads, cost = self.propagate(w,b,X_batch, Y_batch)

        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        params = {"w": w,
                "b": b}
        grads = {"dw": dw,
                "db": db}
        return params, grads, cost

    def fit(self,X,Y, watch=False):
        """
        Parameters
        ------------
        X : 2d array-like (n_samples, n_features)
        Y : 1d array-like (n_samples, )
        
        """
        n_features = X.shape[1]
        n_samples = X.shape[0]
        parameters = self.initialize_parameters(n_features)
        w = parameters['w']
        b = parameters['b']
        X_T = X.T
        Y_T = Y.reshape((1,-1))
        parameters, grads, costs = self.optimize(w, b, X_T, Y_T, self.num_iterations, print_cost=watch)

        self.information['grads'] = grads
        self.information['costs'] = costs
        self.parameters = parameters
        return

    def fit_and_valid(self, X, Y, X_valid, Y_valid, watch=False):
        """
        Parameters
        ------------
        X : 2d array-like (n_samples, n_features)
        Y : 1d array-like (n_samples, )
        """
        n_features = X.shape[1]
        n_samples = X.shape[0]
        parameters = self.initialize_parameters(n_features)
        w = parameters['w']
        b = parameters['b']
        X_T = X.T
        Y_T = Y.reshape((1,-1))
        self.init_mini_batches(X_T, Y_T)
        for i in range(self.num_iterations):
            self.parameters, grads, cost = self.optimize_single(w,b, X_T, Y_T)
            w = self.parameters['w']
            b = self.parameters['b']
            this_loss = self.get_loss(X_valid, Y_valid)
            train_loss = self.get_loss(X, Y)
            self.information['test_loss'].append(this_loss)
            self.information['train_loss'].append(train_loss)
            self.information['cost'].append(cost)
            if i % 50 == 0:
                logger.info('train {}/{}  current cost: {}, train: {} ,test: {}'.format(i,self.num_iterations,cost,train_loss, this_loss))
        self.information['grads'] = grads
        # self.parameters = parameters
        return

    def get_loss(self, X_valid, Y_valid):
        """
        计算这一个样例的相关系数
        """
        Y_valid = Y_valid.reshape(-1)
        y_pred = self.predict(X_valid)
        logger.debug('y_pred : shape{}'.format(y_pred.shape))
        logger.debug('Y_valid : shape{}'.format(Y_valid.shape))
        return precision_score(y_pred, Y_valid)


    def predict(self, X_pred):
        """
        Parameters
        ------------
        X_pred : 2d array-like (n_samples, n_features)

        Retures
        ---------
        Y_pred : 1d array-like (n_samples, )
        """
        n_samples = X_pred.shape[0]
        w = self.parameters['w']
        b = self.parameters['b']
        X = X_pred.T # shape(n_features, n_samples)
        A = sigmoid(np.dot(w.T, X) + b)
        Y_pred = np.zeros(n_samples, dtype=int)
        for i in range(A.shape[1]):
            if A[0,i] > 0.5:
                Y_pred[i] = 1
            elif A[0,i] <= 0.5:
                Y_pred[i] = 0
            else :
                raise NotImplementedError
        return Y_pred