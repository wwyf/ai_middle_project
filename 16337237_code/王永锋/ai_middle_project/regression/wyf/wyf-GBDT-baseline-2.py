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
from pyml.ensemble.regression import GradientBoostingRegression
from pyml.feature_extraction.text import CountVectorizer
from pyml.linear_model.regression import LinearRegression
from pyml.neighbors.classification import KNeighborsClassifier
from pyml.metrics.regression import pearson_correlation
from pyml.model_selection import KFold
from pyml.model_selection import ShuffleSplit
from pyml.preprocessing import StandardScaler
from pyml.logger import logger
import logging
import matplotlib.pyplot as plt

fl = logging.FileHandler('wyf-GBDT-baseline-2.log',mode='a')
formatter = logging.Formatter('[%(levelname)8s] - [%(module)10s] - [%(lineno)3d] - [%(funcName)10s] \n%(message)s\n')
logger.addHandler(fl)

# # 读取数据文件

train = pd.read_excel('../data/train.xlsx')
test = pd.read_excel('../data/testStudent.xlsx')

# 增加里tags特征的属性
train = pd.read_excel('../data/train_add_feat_score.xlsx')
test = pd.read_excel('../data/test_add_feat_score.xlsx')

train.dtypes # 检查有没有数据类型错误的，比如原本是int的变成str，说明里面可能有nan值等奇怪的数据

# train_ori_X = train.drop('Reviewer_Score', axis=1).drop('Tags', axis=1)
# train_ori_Y = train['Reviewer_Score']
# test_ori_X = test.drop('Tags', axis=1)
train_ori_X = train.drop('Reviewer_Score', axis=1)
train_ori_Y = train['Reviewer_Score']
test_ori_X = test

# # 特征工程

def get_proportion_feature_1(df):
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

# 特征方案0：不设置任何特征
train_X_feat = train_ori_X
test_X_feat = test_ori_X

# 特征方案1：增加占比特征，不抛弃原有特征
train_X_feat = get_proportion_feature_1(train_ori_X)
test_X_feat = get_proportion_feature_1(test_ori_X)

train_X_feat.columns

# 查看不同特征与分数的相关系数
for feat_name in train_X_feat:
    print("{} : {}".format(feat_name, pearson_correlation(train_X_feat[feat_name].values, train_ori_Y.values)))

# 方案一：没有权重
ss = StandardScaler()
train_X = ss.fit_transform(train_X_feat.values)
test_X = ss.transform(test_X_feat.values)

# 方案二：设置部分列的权重
ss = StandardScaler()
train_X = ss.fit_transform(train_X_feat.values)
test_X = ss.transform(test_X_feat.values)
# 增加某些特征的权重
train_X[:,1] *= 2
train_X[:,2] *= 2
train_X[:,4] *= 2

train_Y = train_ori_Y.values

# # 交叉验证

logger.setLevel(logging.INFO)

n_splits = 2
k_splits = 10
# cv = ShuffleSplit(n_splits=n_splits)
cv = KFold(k_splits=k_splits)
score = 0
models= []
for train_indices, test_indices in cv.split(train_X):
    lr = GradientBoostingRegression(loss='huber', learning_rate=0.03, n_estimators=120, max_tree_node_size=100)
#     lr.fit(train_X[train_indices], train_Y[train_indices], watch=True)
    lr.fit_and_valid(train_X[train_indices], train_Y[train_indices],train_X[test_indices],train_Y[test_indices], mini_batch=5000 , watch=True)
    y_pred = lr.predict(train_X[test_indices])
    this_score = pearson_correlation(y_pred, train_Y[test_indices])
    score += this_score
    logger.info(this_score)
    models.append(lr)
logger.info('score : {}'.format(score/k_splits))

i = lr
plt.plot(range(len(i.information['test_loss'])),i.information['test_loss'],label='test', )
plt.legend()

for i in models:
    plt.plot(range(len(i.information['test_loss'])),i.information['test_loss'],label='test', )
    plt.legend()

# # 训练模型写入结果

lr = GradientBoostingRegression(learning_rate=0.2, n_estimators=50, max_tree_node_size=500)
lr.fit(train_X, train_Y, watch=True)

y_pred = lr.predict(test_X)
sub = pd.DataFrame(y_pred)
sub.to_csv('./results/'+'GBDT-0.2-50-'+ str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + ".csv", index=0, header=None, index_label=None)

pd.DataFrame(y_pred).plot()

## 一些记录
训练5颗树的时候，验证集大概在0.625-0.640左右
训练10颗树的时候，验证集大概在0.633-0.642左右

# ## 2018.10.17 第二次rank
# 4:GBDT-0.2-50-2018-10-17-13-34
#     1. GBDT
#     2. 特征：无
#     3. 超参数：
#         1. learning_rate=0.2
#         1. n_estimators=50
#         1. max_tree_node_size=500
#     4. 验证集 0.63
#     5. 测试集 53.9545


