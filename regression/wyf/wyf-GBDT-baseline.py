# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   jupytext_formats: py:light
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.5
# ---

import sys
# sys.path.append(r'E:/0code')
sys.path.append(r'/home/wyf/0code')
# sys.path.append(r'/home/wangyf226/0code')
# sys.path.append(r'/BIGDATA1/nsccgz_yfdu_1/asc19/wyf/pyml')

# %load_ext autoreload
# %autoreload 2
import pandas as pd
import numpy as np
import datetime
from pyml.emsemble.regression import GradientBoostingRegression
from pyml.feature_extraction.text import CountVectorizer
from pyml.linear_model.regression import LinearRegression
from pyml.neighbors.classification import KNeighborsClassifier
from pyml.metrics.regression import pearson_correlation
from pyml.model_selection import KFold
from pyml.model_selection import ShuffleSplit
from pyml.preprocessing import StandardScaler

# # 读取数据文件

train = pd.read_excel('../data/train.xlsx')
test = pd.read_excel('../data/testStudent.xlsx')

train.dtypes # 检查有没有数据类型错误的，比如原本是int的变成str，说明里面可能有nan值等奇怪的数据

train_ori_X = train.drop('Reviewer_Score', axis=1).drop('Tags', axis=1)
train_ori_Y = train['Reviewer_Score']
test_ori_X = test.drop('Tags', axis=1)

# # 特征工程

def get_proportion_feature(df):
    """
    构造以下三个特征
    积极评论占总评论的比例
    消极评论占总评论的比例
    评论员评论占总评论的比例
    """
    df = df.copy()
    
    base_features = ['Total_Number_of_Reviews']
    gap_features = ['Review_Total_Negative_Word_Counts', 'Review_Total_Positive_Word_Counts','Total_Number_of_Reviews_Reviewer_Has_Given']
    for base_feature in base_features:
        for gap_feature in gap_features:
            df[gap_feature+'_radio_'+base_feature] = df[gap_feature]/df[base_feature]
            # 数字太小了，乘上一个10
#             df = df.drop(gap_feature, axis=1)
    return df

# # 构造训练集和测试集，并归一化

train_X_feat = get_proportion_feature(train_ori_X)
test_X_feat = get_proportion_feature(test_ori_X)

train_X_feat.columns

ss = StandardScaler()
train_X = ss.fit_transform(train_X_feat.values)
test_X = ss.transform(test_X_feat.values)

train_Y = train_ori_Y.values

# # 交叉验证

n_splits = 3
cv = ShuffleSplit(n_splits=n_splits)
for train_indices, test_indices in cv.split(train_X):
    lr = GradientBoostingRegression(learning_rate=0.1, n_estimators=100, max_tree_node_size=400)
    lr.fit(train_X[train_indices], train_Y[train_indices], watch=True)
    y_pred = lr.predict(train_X[test_indices])
    print(pearson_correlation(y_pred, train_Y[test_indices]))

# # 训练模型写入结果

lr = GradientBoostingRegression(learning_rate=0.1, n_estimators=100, max_tree_node_size=400)
lr.fit(train_X, train_Y, watch=True)

y_pred = lr.predict(test_X)
sub = pd.DataFrame(y_pred)
sub.to_csv('../results/'+'GBDT-0.1-100-'+ str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + ".csv", index=0, header=None, index_label=None)


