import numpy as np
import json

from collections import Counter
def find_most_common(l):
    """
    寻找l列表中出现次数最多的元素
    """
    label_count = Counter(l)
    most_common_label = label_count.most_common(1)[0][0]
    return most_common_label

def entropy(l):
    """
    Parameters
    ----------
    l : 1d array-like shape(n_samples, )

    """
    feature_values, counts = np.unique(l, return_counts=True)
    probabilities = counts/counts.sum()
    l_h = (-probabilities*np.log2(probabilities)).sum()
    return l_h

def condition_entropy(left, right):
    """
    Parameters
    ----------
    left : 1d array-like shape(n_samples, )
        条件对应的列
    right : 1d array-like shape(n_samples, )
        用来计算熵的一列 

    Returns
    ------------
    result : condition entropy
    """
    feature_values, counts = np.unique(left, return_counts=True)
    probabilities = counts/counts.sum()
    entropies = np.zeros((len(feature_values)))
    for i,feature_value in enumerate(feature_values):
        # 对于每一个value
        # 先找出对应的y中的所有索引，取出相关的y
        # 然后调用计算熵的函数去计算
        this_indices = np.argwhere(left == feature_value).reshape(-1)
        entropies[i] = entropy(right[this_indices])
    result = (probabilities * entropies).sum()
    return result

def information_gain(left, right):
    """
    计算特征 left 对于 数据集right的信息增益

    Parameters
    ------------
    left : 1d array-like shape(n_samples, )
        条件对应的列
    right : 1d array-like shape(n_samples, )
        用来计算熵的一列 

    Returns
    ----------
    result : information_gain

    """
    return entropy(right) - condition_entropy(left, right)

def information_gain_radio(left, right):
    """
    计算特征 left 对于 数据集right的信息增益率

    Parameters
    ------------
    left : 1d array-like shape(n_samples, )
        条件对应的列
    right : 1d array-like shape(n_samples, )
        用来计算熵的一列 

    Returns
    ----------
    result : information_gain_radio

    """
    split_info = entropy(left)
    infor_gain = information_gain(left, right)
    # FIXME: 可能信息量为0所以加1，这一个需要吗？
    return infor_gain/(split_info+0.00000001)


def gini(l):
    """
    计算特征 left 对于 数据集right的gini指数

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


def condition_gini(left, right):
    """
    Parameters
    ----------
    left : 1d array-like shape(n_samples, )
        条件对应的列
    right : 1d array-like shape(n_samples, )
        用来计算熵的一列 

    Returns
    ------------
    result : condition entropy
    """
    feature_values, counts = np.unique(left, return_counts=True)
    probabilities = counts/counts.sum()
    entropies = np.zeros((len(feature_values)))
    for i,feature_value in enumerate(feature_values):
        # 对于每一个value
        # 先找出对应的y中的所有索引，取出相关的y
        # 然后调用计算熵的函数去计算
        this_indices = np.argwhere(left == feature_value).reshape(-1)
        entropies[i] = gini(right[this_indices])
    result = (probabilities * entropies).sum()
    return result

def gini_gain(left, right):
    """
    计算特征 left 对于 数据集right的gini信息增益

    Parameters
    ------------
    left : 1d array-like shape(n_samples, )
        条件对应的列
    right : 1d array-like shape(n_samples, )
        用来计算熵的一列 

    Returns
    ----------
    result : information_gain

    """
    return gini(right) - condition_gini(left, right)


class DecisionTreeClassifier():
    """
    寻找 向量[x1, x2, x3, ... , xn] -> 种类 y 的映射关系

    Parameters
    ----------
    method : string
        - 'id3'
        - 'c4.5'
        - 'cart'

    Notes
    ------
    None
    """
    def __init__(self, method='id3'):
        self.method = method

    def build_tree(
            self,ori_X,
            sub_X, sub_Y, features,
            parent_class=None
    ):
        """
        Parameters
        ------------
        sub_X : 2d array-like shape(n_samples, n_features)
        sub_Y : 1d array-like shape(n_samples, )
        features : list of string
            特征名的列表，表示当前尚未被使用来建树的特征

        Returns
        -----------
        tree : dict
        """
        # 如果此时数据集为空 即此时子节点所获得的数据集中某特征下的量并不完整
        if len(sub_X) == 0:
            return parent_class.item()
        # 有这些情况是可能需要直接返回值的，表示已经能够确定分类结果
        # 此时的数据集都分到了同一类，不需要再分类
        if len((np.unique(sub_Y))) <= 1:
            return sub_Y[0].item()

        # 如果此时深度过大，所有特征已经被用完了
        if len(features) == 0:
            return parent_class.item()
        

        # 从sub_Y里面取出现次数最多的，作为该节点的结果
        current_node_class = np.unique(sub_Y)[np.argmax(np.unique(sub_Y, return_counts=True)[1])]

        # 选取一个特征用于划分数据集
        if (self.method == 'id3'):
            score_func = information_gain
        elif self.method == 'c4.5':
            score_func = information_gain_radio
        elif self.method == 'cart':
            score_func = gini_gain
        else :
            raise NotImplementedError
        feature_values = [score_func(sub_X[:,self.get_feature_columns_index(feature)], sub_Y) for feature in features]
        best_feature_index = np.argmax(feature_values)
        best_feature = features[best_feature_index]

        # 如果子数据集中的所有样本的所有特征都相同，此时无法划分
        # 将当前节点标记为叶节点，类别为D中出现次数最多的类
        if feature_values[best_feature_index] == 0:
            return current_node_class.item()

        tree = {best_feature:{}}

        # 将best feature删掉
        features = [i for i in features if i != best_feature]
        best_feature_index = self.get_feature_columns_index(best_feature)
        # NOTE: 这里改成了ori_X 
        # 是为了能够让特征无论建在树的哪一层都能够知道整个数据集该特征的所有取值
        for value in np.unique(ori_X[:,best_feature_index]):
            # 取一个子数据集
            sub_sub_X = sub_X[sub_X[:, best_feature_index] == value]
            sub_sub_Y = sub_Y[sub_X[:, best_feature_index] == value]

            # 生成新树
            subtree = self.build_tree(ori_X, sub_sub_X, sub_sub_Y, features, current_node_class)

            # 将树节点加到根节点下
            tree[best_feature][value] = subtree
        
        return tree

    def fit(self, X, Y, feature_names=None):
        """

        Parameters
        -----------
        X : 2d array-like shape(n_samples, n_features)

        Y : 1d array-like shape(n_samples,) not-negative number
        
        feature_names : list of string shape(n_features) optional
            表示X中每一列的含义，如果有的话，能够有一颗意义更加清晰的决策树，没有也不要紧

        """
        if feature_names is None:
            feature_names = range(0, X.shape[1])
        assert(X.shape[1] == len(feature_names))
        self.feature_names = feature_names
        self.root_node = self.build_tree(X, X, Y, feature_names, self.method)

    def get_feature_columns_index(self, feature_name):
        return self.feature_names.index(feature_name)

    def _predict_subtree(self, x_dict, tree):
        for key in list(x_dict.keys()):
            # 遍历X的feature name ,如果在决策树当前根节点中找到了对应的feature，就尝试返回该feature value下对应的结果，如果该结果仍然是一棵树，就继续递归循环
            if key in list(tree.keys()):
                # print(json.dumps(tree,indent=4,sort_keys=True))
                # FIXME[FIXED]: 不知道为什么，树可能是不完备的？ 已FIX
                # try :
                #     result = tree[key][x_dict[key]]
                # except :
                #     return 1
                result = tree[key][x_dict[key]]
                if isinstance(result, dict):
                    return self._predict_subtree(x_dict, result)
                else :
                    return result

    def _predict_one(self, x):
        """

        Parameters
        -----------
        x : 1d array-like shape(n_features, )

        Returns
        ----------

        result : int or str
            在决策树中的分类结果
        """
        # return self.root_node.get_classification_result(x)

        x_dict = {}
        for i, key in enumerate(self.feature_names):
            x_dict[key] = x[i]
        result = self._predict_subtree(x_dict, self.root_node)
        return result

    def predict(self, X_pred):
        """

        Parameters
        -------------
            X_pred : 2d array-like shape(n_samples, n_feature)

        Returns
        --------------
            Y_pred : 1d array-like shape(n_samples,)

        """
        Y_pred = np.zeros((X_pred.shape[0]))
        for i,x in enumerate(X_pred):
            Y_pred[i] = self._predict_one(x)
        return Y_pred
    
    def print_tree_graph(self):
        print(json.dumps(self.root_node,indent=4,sort_keys=True))