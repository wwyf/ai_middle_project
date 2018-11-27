import numpy as np
from pydot import Dot, Edge, Node
from pyml.logger import logger

# 这里要实现CART树

# 1. 如何选择分裂点？
#     1. 对于离散型特征：遍历特征中的每一个值
#         1. 创建两个节点，左：样例特征等于该值 右：样例特征不等于该值
#         2. 计算分裂成本：分类任务中，寻找gini值最小的，回归任务中，寻找平方误差最小的
#     2. 对于连续性数据：先对数据进行排序， 遍历每一个可能的分裂点
#         1. 创建一个分裂点，左：小于这个分裂点的数据 右：大于这个分裂点的数据
#         2. 计算分裂成本：同上

# CART回归树，如何更robust？

def generate_counter(counter=0):
    while True:
        yield counter
        counter += 1

generate_id = generate_counter()

def gini(l):
    """
    Parameters
    ------------
    l : 1d array-like shape(n_samples, )
        条件对应的列

    Returns
    ----------
    result : gini
    """
    feature_values, counts = np.unique(l, return_counts=True)
    probabilities = counts/counts.sum()
    l_h = 1-(np.square(probabilities)).sum()
    return l_h

class CartTreeClassifierNode():
    def __init__(self, feature_names : list, column_flags : list, max_node_size=10, divide_way='half', cost_func=gini):
        """
        feature_names : list of string
            就是特征名字的列表啦，与矩阵的列号对应，一直都不变
        column_flags : list of string
            'discrete','continuous'
        cost_func : cost function
        """
        self.id = next(generate_id)
        self.max_node_size = max_node_size
        self.feature_names = feature_names
        self.column_flags = column_flags
        self.cost_func = gini
        self.is_leaf = False
        self.feature_column = None # 数据集划分点所在特征的列号
        self.split_op = None # 数据集划分所使用标志，离散用‘==’，连续用‘<='
        self.split_value = None # 数据集划分使用该特征的值
        self.left_tree = None
        self.right_tree = None
        self.current_node_class = None
        self.divide_way = divide_way
        logger.debug('self.id : {}'.format(self.id) +
                    '\nself.max_node_size : {}'.format(self.max_node_size) +
                    '\nself.feature_names : {}'.format(self.feature_names) +
                    '\nself.column_flags : {}'.format(self.column_flags))
    def fit_data(self, sub_X, sub_Y, parent_class):
        """
        sub_X : 2d array-like shape(n_samples, n_features)
        sub_Y : 1d array-like shape(n_samples )
        paranet_class : int
            递归过程中，要先记一下父节点的属性，可能有用
        """

        logger.debug('training...\ncurrent id : {}\ncurrent data size : {}'.format(self.id, sub_X.shape[0]))

        logger.debug(
            'X : \n{}\nY : {}'.format(sub_X, sub_Y)+'\n'+
            'parent_class : {}'.format(parent_class))

        # 如果此时数据集为空
        if len(sub_X) == 0:
            logger.debug('sub_X is empty ! ')
            self.set_leaf(parent_class.item())
            return
        # 如果此时分类的Y都是类似的，已经能够确定分类结果
        if len((np.unique(sub_Y))) <= 1:
            logger.debug('sub_Y is all the same ! ')
            self.set_leaf((sub_Y[0].item()))
            return

        # 从sub_Y里面取出现次数最多的，作为该节点的结果
        self.current_node_class = np.unique(sub_Y)[np.argmax(np.unique(sub_Y, return_counts=True)[1])]

        logger.debug('self.current_node_class : {}'.format(self.current_node_class))

        if sub_X.shape[0] <= self.max_node_size:
            logger.debug('sub_X is so small. n_samples : {}'.format(sub_X.shape[0]))
            self.set_leaf(self.current_node_class)
            return

        # 寻找数据集的最佳拆分点，寻找best_feature_name, best_feature_split_name
        # 并且顺便将数据拆成左右两个分支
        best_gini_value = 9999999
        best_feature_column = None
        best_split_point = None

        logger.debug('in find the best split feature...')

        for this_feature_index in range(0,len(self.feature_names)):
            n_samples = sub_X.shape[0]
            if self.divide_way == 'default':
                this_feature_values = np.unique(sub_X[:, this_feature_index])
                # 获得当前feature最佳split_point的gini指数，并与当前最佳比较
                # 得到这一个feature的所有可取的值
                for this_feature_value in this_feature_values:
                    # 对可能取到的每一个值，我都要计算以这个值为分割点时的gini指数
                    if self.column_flags[this_feature_index] == 'continuous':
                        # 如果当前列的flag说明是连续的
                        # TODO: 似乎不能够取到最右的端点,那如果等于直接跳过吧~
                        if this_feature_value == np.amax(sub_X[:,this_feature_index]):
                            continue
                        left_branch_Y = sub_Y[sub_X[:,this_feature_index]<=this_feature_value]
                        right_branch_Y = sub_Y[sub_X[:,this_feature_index]>this_feature_value]
                    elif self.column_flags[this_feature_index]  == 'discrete':
                        # 如果当前列的flag说明是离散的
                        left_branch_Y = sub_Y[sub_X[:,this_feature_index]==this_feature_value]
                        right_branch_Y = sub_Y[sub_X[:,this_feature_index]!=this_feature_value]
                    this_feature_gini_value = len(left_branch_Y)/n_samples * gini(left_branch_Y) + len(right_branch_Y)/n_samples * gini(right_branch_Y)

                    logger.debug('in feature({}:{}) value({}) gini_value({})\nleft_branch_Y : {}\nright_branch_Y : {}'.format(this_feature_index,self.feature_names[this_feature_index], this_feature_value, this_feature_gini_value, left_branch_Y, right_branch_Y))

                    # 如果以这个值为分割点的gini指数更小，那就更新best参数
                    if this_feature_gini_value < best_gini_value:
                        best_gini_value = this_feature_gini_value
                        best_feature_column = this_feature_index
                        best_split_point = this_feature_value
            elif self.divide_way == 'half':
                #  这一个就是，对于连续数据直接对半分
                if self.column_flags[this_feature_index] == 'continuous':
                    this_feature_value = (np.max(sub_X[:, this_feature_index])+np.min(sub_X[:, this_feature_index]))/2
                    left_branch_Y = sub_Y[sub_X[:,this_feature_index]<=this_feature_value]
                    right_branch_Y = sub_Y[sub_X[:,this_feature_index]>this_feature_value]
                    this_feature_gini_value = len(left_branch_Y)/n_samples * gini(left_branch_Y) + len(right_branch_Y)/n_samples * gini(right_branch_Y)
                    logger.debug('in feature({}:{}) value({}) gini_value({})\nleft_branch_Y : {}\nright_branch_Y : {}'.format(this_feature_index,self.feature_names[this_feature_index], this_feature_value, this_feature_gini_value, left_branch_Y, right_branch_Y))
                    # 如果以这个值为分割点的gini指数更小，那就更新best参数
                    if this_feature_gini_value < best_gini_value:
                        best_gini_value = this_feature_gini_value
                        best_feature_column = this_feature_index
                        best_split_point = this_feature_value
                elif self.column_flags[this_feature_index]  == 'discrete':
                    # 离散数据
                    this_feature_values = np.unique(sub_X[:, this_feature_index])
                    for this_feature_value in this_feature_values:
                        n_samples = sub_X.shape[0]
                        left_branch_Y = sub_Y[sub_X[:,this_feature_index]==this_feature_value]
                        right_branch_Y = sub_Y[sub_X[:,this_feature_index]!=this_feature_value]
                        this_feature_gini_value = len(left_branch_Y)/n_samples * gini(left_branch_Y) + len(right_branch_Y)/n_samples * gini(right_branch_Y)
                        logger.debug('in feature({}:{}) value({}) gini_value({})\nleft_branch_Y : {}\nright_branch_Y : {}'.format(this_feature_index,self.feature_names[this_feature_index], this_feature_value, this_feature_gini_value, left_branch_Y, right_branch_Y))
                        # 如果以这个值为分割点的gini指数更小，那就更新best参数
                        if this_feature_gini_value < best_gini_value:
                            best_gini_value = this_feature_gini_value
                            best_feature_column = this_feature_index
                            best_split_point = this_feature_value
            else:
                raise NotImplementedError
        
        self.feature_column = best_feature_column
        self.split_value = best_split_point
        logger.debug('get the best split point : {}:{}/{}'.format(self.feature_column, self.feature_names[self.feature_column], self.split_value))

        # print('self.feature_column:', self.feature_column)
        # print('self.split_value', self.split_value)
        # c=input()

        # 划分数据集
        if self.column_flags[this_feature_index] == 'continuous':
            # 如果当前列的flag说明是连续的
            # 取得最好的数据集拆分方式
            self.split_op = '<='
            self.split_value = best_split_point
            best_left_branch_X = sub_X[sub_X[:,best_feature_column]<=best_split_point,:]
            best_left_branch_Y = sub_Y[sub_X[:,best_feature_column]<=best_split_point]
            best_right_branch_X = sub_X[sub_X[:,best_feature_column]>best_split_point,:]
            best_right_branch_Y = sub_Y[sub_X[:,best_feature_column]>best_split_point]
        elif self.column_flags[this_feature_index]  == 'discrete':
            # 如果当前列的flag说明是离散的
            self.split_op = '=='
            self.split_value = best_split_point
            best_left_branch_X = sub_X[sub_X[:,best_feature_column]==best_split_point,:]
            best_left_branch_Y = sub_Y[sub_X[:,best_feature_column]==best_split_point]
            best_right_branch_X = sub_X[sub_X[:,best_feature_column]!=best_split_point,:]
            best_right_branch_Y = sub_Y[sub_X[:,best_feature_column]!=best_split_point]

        logger.debug('get left branch X : \n{}\nget left branch Y : {}'.format(best_left_branch_X, best_left_branch_Y))
        logger.debug('get right branch X : \n{}\nget right branch Y : {}'.format(best_right_branch_X, best_right_branch_Y))

        self.left_tree = CartTreeClassifierNode( self.feature_names, self.column_flags, max_node_size=self.max_node_size, divide_way=self.divide_way , cost_func=self.cost_func)
        self.left_tree.fit_data(best_left_branch_X, best_left_branch_Y, self.current_node_class)
        self.right_tree = CartTreeClassifierNode(self.feature_names, self.column_flags, max_node_size=self.max_node_size, divide_way=self.divide_way , cost_func=self.cost_func)
        self.right_tree.fit_data(best_right_branch_X, best_right_branch_Y, self.current_node_class)

    def set_leaf(self, current_class):
        self.is_leaf = True
        self.current_node_class = current_class

    def is_left(self, x):
        """

        每一个具有两个子树的节点，必然经历了建树的过程，初始化了op与value
        因此需要提供，某向量，是否要在左子树中搜索的判断函数
        判断该向量是否要在该节点的左子树中继续寻找，如果否，则是要在右子树中继续寻找 

        Parameters
        ------------
        x : 1d array shape(n_feature, )

        Requirement
        -------------
        self.feature_column
        self.feature_name
        self.op
        self.value
        """
        x = x.reshape((1,-1))
        if self.split_op == '<=':
            return x[0,self.feature_column] <= self.split_value
        elif self.split_op == '==':
            return x[0,self.feature_column] == self.split_value
        else:
            raise NotImplementedError
    def get_label(self, x):
        if self.is_leaf:
            return self.current_node_class
        elif self.is_left(x):
            return self.left_tree.get_label(x)
        else :
            return self.right_tree.get_label(x)
    
    def print_tree(self, filename):
        graph = Dot(graph_type='digraph')
        full_graph = self._print_tree(graph)
        full_graph.write_png(filename)
    
    def get_node_str(self):
        if self.is_leaf:
            return "id : {}\n label: {}".format(self.id,self.current_node_class)
        else:
            return 'id : {}\nfesture : {}/{}\n{} {}'.format(self.id, self.feature_column, self.feature_names[self.feature_column], self.split_op, self.split_value)
    
    def _print_tree(self,graph):
        root = self.get_node_str()
        if self.is_leaf:
            return graph
        left_tree_str = self.left_tree.get_node_str()
        right_tree_str = self.right_tree.get_node_str()
        # print('left', left_tree_str)
        # print('right', right_tree_str)

        graph.add_edge(Edge(src=root, dst=left_tree_str))
        graph = self.left_tree._print_tree(graph)
        graph.add_edge(Edge(src=root, dst=right_tree_str))
        graph = self.right_tree._print_tree(graph)
        return graph


class DecisionTreeClassifier():
    """
    使用cart方法，二叉树
    TODO: 没有写剪枝
    """
    def __init__(self, max_node_size=10, divide_way='half'):
        self.max_node_size = max_node_size
        self.divide_way = divide_way

    def fit(self, X, Y, column_flags, feature_names=None):
        """
        Parameters
        -----------
        X : 2d array-like
        Y : 1d array-like
        """
        logger.debug('X : \n{}'.format(X))
        logger.debug('Y : {}'.format(Y))
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [str(i) for i in range(n_features)]
        logger.debug('feature_names : {}'.format(feature_names))
        self.root_node = CartTreeClassifierNode(
            feature_names, 
            column_flags, 
            max_node_size=self.max_node_size,
            divide_way=self.divide_way
        )
        self.root_node.fit_data(X, Y,None)

    def predict(self, X_pred):
        """
        Parameters
        ----------
        X_pred : 2d array-like shape(n_samples, n_features)

        Returns
        -------
        Y_pred : 1d array-like shape(n_samples, )

        """
        n_samples = X_pred.shape[0]
        Y_pred = np.zeros((n_samples))
        for i,x in enumerate(X_pred):
            Y_pred[i] = self.root_node.get_label(x)
        return Y_pred

if __name__ == '__main__':
    a = np.array([[1,4, 0.4],
                  [2,9, 0.6],
                  [3,6, 0.8],
                  [1,7, 0.2],
                  [1,8, 0.46]])
    b = np.array([0,0,1,1,1])
    flags = ['discrete','continuous','continuous']
    names = ['feat1','feat2','feat3']
    clf = DecisionTreeClassifier(max_node_size=1)
    clf.fit(a,b,flags,names)
    clf.root_node.print_tree('test.png')
    print(clf.predict(np.array([[2,5, 0.6]])))
    print(clf.predict(np.array([[1,7,0.2]])))
    print(clf.predict(np.array([[1,5, 0.8]])))