import numpy as np
import matplotlib.pyplot as plt
from pyml.logger import logger
from pyml.metrics.classification import precision_score
from pyml.preprocessing import StandardScaler
import math


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def nn_model_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.random.randn(1, 3)
    return X_assess, Y_assess

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


class MLPClassifier():
    def __init__(self, learning_rate=1.2, hidden_size=4, num_iterations=10000, optimizer='gd', mini_batch = 64 , random_state = 3, print_intervel=10):
        np.random.seed(random_state)
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations
        self.parameters = {}
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.mini_batch = mini_batch
        self.information = {
            'train_loss' : [],
            'valid_loss' : [],
            'cost' : []
        }
        self.mini_batches = []
        self.current_mini_batch_index = 0
        self.print_intervel = print_intervel
        # self.train_X
        # self.train_Y
        # self.valid_X
        # self.valid_Y
    def initialize_parameters(self, n_x, n_h, n_y):
        """
        初始化参数
        """
        np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
        W1 = np.random.randn(n_h,n_x) * 0.01
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h) * 0.01
        b2 = np.zeros((n_y,1))
        
        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))
        
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        logger.debug('initialize_parameters : {}'.format(parameters))
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
        return current_mini_batch
    def update_mini_batch(self):
        self.current_mini_batch_index = (self.current_mini_batch_index+1) % len(self.mini_batches)

    def feat_data(self, X_train, Y_train, X_valid=None, Y_valid=None):
        """
        将数据弄进去，构建输入层，隐藏层，输出层
        Parameters
        ------------
        X_train : shape(n_samples, n_features)
        Y_train : shape(n_samples, )
        X_valid : shape(n_samples, n_features)
        Y_valid : shape(n_samples, )
        """
        X_train = X_train.T
        Y_train = Y_train.reshape((1,-1))
        self.init_mini_batches(X_train, Y_train)
        if X_valid is not None:
            X_valid = X_valid.T
        if Y_valid is not None:
            Y_valid = Y_valid.reshape((1,-1))
        n_x = X_train.shape[0] # size of input layer
        n_h = self.hidden_size
        n_y = Y_train.shape[0] # size of output layer
        self.structure = (n_x, n_h, n_y)
        logger.debug('self.structure : {}'.format(self.structure))
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.parameters = self.initialize_parameters(n_x, n_h, n_y)
        logger.debug('self.X_train : \n{}\nshape : {}'.format(self.X_train, self.X_train.shape))
        logger.debug('self.Y_train : \n{}\nshape : {}'.format(self.Y_train, self.Y_train.shape))

    def forward(self, X_test=None, predict=False):
        """
        前向传播
        X : shape(n_x, n_samples)

        Returns
        -----------
        A2 : shape(1, n_samples)
        """
        if predict is True:
            X = X_test
        elif self.optimizer == 'mini-batch':
            X = self.get_mini_batch()[0]
        else:
            X = self.X_train # TODO: 
        # Retrieve each parameter from the dictionary "parameters"
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        # logger.debug('W1.shape : {}'.format(W1.shape))
        # logger.debug('b1.shape : {}'.format(b1.shape))
        # logger.debug('W2.shape : {}'.format(W2.shape))
        # logger.debug('b2.shape : {}'.format(b2.shape))
        # Implement Forward Propagation to calculate A2 (probabilities)
        Z1 = np.dot(W1,X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2,A1) + b2
        A2 = sigmoid(Z2)
        assert(A2.shape == (1, X.shape[1]))
        if predict is False:
            self.cache = {"Z1": Z1,
                    "A1": A1,
                    "Z2": Z2,
                    "A2": A2}
        # logger.debug('A2 : \n{}'.format(A2))
        return A2

    def backward(self):
        """
        计算梯度
        """
        if self.optimizer == 'mini-batch':
            X,Y = self.get_mini_batch()
        else:
            X,Y = self.X_train, self.Y_train
        m = X.shape[1]
        # First, retrieve W1 and W2 from the dictionary "parameters".
        W1 = self.parameters["W1"]
        W2 = self.parameters["W2"]
        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = self.cache["A1"]
        A2 = self.cache["A2"]
        # Backward propagation: calculate dW1, db1, dW2, db2. 
        dZ2= A2 - Y
        dW2 = 1 / m * np.dot(dZ2,A1.T)
        db2 = 1 / m * np.sum(dZ2,axis=1,keepdims=True)
        dZ1 = np.dot(W2.T,dZ2) * (1-np.power(A1,2))
        dW1 = 1 / m * np.dot(dZ1,X.T)
        db1 = 1 / m * np.sum(dZ1,axis=1,keepdims=True)
        self.grads = {"dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2}
        logger.debug('grads : \n{}'.format(self.grads))
        return

    def update_parameters(self, learning_rate = 1.2):
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        dW1 = self.grads["dW1"]
        db1 = self.grads["db1"]
        dW2 = self.grads["dW2"]
        db2 = self.grads["db2"]
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        self.parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        return

    def step(self):
        """
        使用梯度更新参数
        """
        if self.optimizer == 'gd':
            self.update_parameters(self.learning_rate)
        elif self.optimizer == 'mini-batch':
            self.update_parameters(self.learning_rate)
        else:
            raise NotImplementedError

    def compute_cost(self):
        """
        Parameters
        -------------
        A2 : shape(1, n_samples)
        Y : shape(1, n_samples)
        """
        A2 = self.cache['A2']
        if self.optimizer == 'mini-batch':
            Y = self.get_mini_batch()[1]
        else:
            Y = self.Y_train

        m = Y.shape[1] # number of example

        # Compute the cross-entropy cost
        logprobs = Y*np.log(A2) + (1-Y)* np.log(1-A2)
        cost = -1/m * np.sum(logprobs)
        cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
        assert(isinstance(cost, float))
        return cost
    def get_train_and_valid_result(self):
        """
        Returns
        -------------
        train_loss : float
        valid_loss : float
        """
        Y_train_pred = np.round(self.forward(X_test=self.X_train, predict=True)).reshape((-1))
        logger.debug('Y_train_pred : \n{}'.format(Y_train_pred))
        logger.debug('self.Y_train : \n{}'.format(self.Y_train))
        train_loss = precision_score(Y_train_pred, self.Y_train.reshape(-1))

        if self.X_valid is not None:
            Y_valid_pred = np.round(self.forward(X_test=self.X_valid, predict=True))
            Y_valid_pred.reshape((-1))
            logger.debug('Y_valid_pred : \n{}'.format(Y_valid_pred))
            valid_loss = precision_score(Y_valid_pred, self.Y_valid.reshape(-1))
        else:
            valid_loss = 0

        return train_loss, valid_loss

    def predict(self, X_test):
        """
        Parameters
        --------------
        X_test : shape(n_samples, n_features)
        """
        A2 = self.forward(X_test=X_test.T,predict=True)
        logger.debug('A2 : \n{}'.format(A2))
        return np.round(A2).reshape(-1)
    
    def train(self):
        for i in range(self.num_iterations):
            cost, train_loss, valid_loss = self.train_one()
            if i % self.print_intervel == 0:
                logger.info('train {}/{}  current cost: {}, train: {} ,valid: {}'.format(i,self.num_iterations,cost,train_loss, valid_loss))

    def train_one(self):
        self.forward()
        self.backward()
        self.step()
        cost = self.compute_cost()
        train_loss, valid_loss = self.get_train_and_valid_result()
        self.information['train_loss'].append(train_loss)
        self.information['valid_loss'].append(valid_loss)
        self.information['cost'].append(cost)
        self.update_mini_batch()
        return cost,train_loss, valid_loss


def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    return X, Y


if __name__ == '__main__':
    logger.setLevel(20)
    # X_assess, Y_assess = nn_model_test_case()
    # logger.debug('X_assess : \n{}\nshape : {}'.format(X_assess, X_assess.shape))
    # logger.debug('Y_assess : \n{}\nshape : {}'.format(Y_assess, Y_assess.shape))
    # clf = MLPClassifier(hidden_size=4, num_iterations=10000)
    # clf.feat_data(X_assess.T, Y_assess.T)
    # clf.train()
    # print("W1 = " + str(clf.parameters["W1"]))
    # print("b1 = " + str(clf.parameters["b1"]))
    # print("W2 = " + str(clf.parameters["W2"]))
    # print("b2 = " + str(clf.parameters["b2"]))
# *********************************
    # mini_train_X = np.array([
    #     [1,2,3,4,5,6,7,8],
    #     [2,3,4,5,6,7,8,9],
    #     [3,4,5,6,7,8,9,10],
    #     [4,5,6,7,8,9,10,11],
    #     [5,6,7,8,9,10,11,12],
    #     [6,7,8,9,10,11,12,13],
    #     [7,8,9,10,11,12,13,14]
    # ])
    # mini_train_Y = np.array([
    #     0,1,0,1,0,1,0
    # ])
    # mini_test_X = np.array([
    #     [2,3,4,5,6,7.5,8,9],
    #     [4,5,6,7.5,8,9,10,11]
    # ])
    # mini_standard_out_Y = np.array([
    #     1,0
    # ])
    # ss = StandardScaler()
    # mini_train_X = ss.fit_transform(mini_train_X)
    # mini_test_X = ss.transform(mini_test_X)
    # print(mini_train_X)
    # print(mini_test_X)
    # clf = MLPClassifier(hidden_size=4, num_iterations=1000, learning_rate=1.2)
    # clf.feat_data(mini_train_X, mini_train_Y, mini_test_X, mini_standard_out_Y)
    # clf.train()
    # print(clf.predict(mini_test_X))
#*************************************
    X,Y = load_planar_dataset()
    clf = MLPClassifier(hidden_size=4, num_iterations=10000)
    clf.feat_data(X, Y, X, Y)
    clf.train()
    print(clf.parameters)
    print(clf.predict(X))
    print(load_planar_dataset())