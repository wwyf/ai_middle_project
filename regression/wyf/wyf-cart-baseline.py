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
from pyml.tree.regression import DecisionTreeRegressor
from pyml.feature_extraction.text import CountVectorizer
from pyml.linear_model.regression import LinearRegression
from pyml.neighbors.classification import KNeighborsClassifier
from pyml.metrics.regression import pearson_correlation
from pyml.model_selection import KFold
from pyml.model_selection import ShuffleSplit
from pyml.preprocessing import StandardScaler
from pyml.logger import logger
import logging

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

drop_feature_names = ['Total_Number_of_Reviews_Reviewer_Has_Given', 'with_pet_score']
for drop_feature_name in drop_feature_names:
    train_X_feat = train_X_feat.drop(labels=drop_feature_name, axis=1)
    test_X_feat = test_X_feat.drop(labels=drop_feature_name, axis=1)

# 查看不同特征与分数的相关系数
for feat_name in train_X_feat:
    print("{} : {}".format(feat_name, pearson_correlation(train_X_feat[feat_name].values, train_ori_Y.values)))

# +
# 归一化，可选择不同方案
# -

# 方案一：没有权重
ss = StandardScaler()
train_X = ss.fit_transform(train_X_feat.values)
test_X = ss.transform(test_X_feat.values)

# 方案二：设置部分列的权重
ss = StandardScaler()
train_X = ss.fit_transform(train_X_feat.values)
test_X = ss.transform(test_X_feat.values)
# 增加某些特征的权重
train_X[:,1] *= 4
train_X[:,2] *= 4
train_X[:,4] *= 2
train_X[:,5] *= 2
train_X[:,9] *= 3

train_Y = train_ori_Y.values

# # 交叉验证

logger.setLevel(logging.INFO)

n_splits = 5
cv = ShuffleSplit(n_splits=n_splits)
for train_indices, test_indices in cv.split(train_X):
#     lr = GradientBoostingRegression(learning_rate=0.1, n_estimators=100, max_tree_node_size=400)
    lr = DecisionTreeRegressor(max_node_size=1000)
#     lr.fit(train_X[train_indices], train_Y[train_indices], watch=True)
    lr.fit(train_X[train_indices], train_Y[train_indices])
    y_pred = lr.predict(train_X[test_indices])
    logger.info(pearson_correlation(y_pred, train_Y[test_indices]))

# # 训练模型写入结果

lr = DecisionTreeRegressor(max_node_size=1000)
lr.fit(train_X, train_Y)

y_pred = lr.predict(test_X)
sub = pd.DataFrame(y_pred)
sub.to_csv('./results/'+'CART-m1000-no_weight-add_tag_feat'+ str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + ".csv", index=0, header=None, index_label=None)





# # 记录提交结果
# ## 2018.10.17 第一次rank
# 1： CART-m500-2018-10-17-10-34.csv : 
#     1. 模型：CART二叉回归树
#     2. 特征：增加特征1，并抛弃评论词汇数量
#     2. 超参数：max_node_size=500,没有设置连续值特征搜索上限 ：
#     3. 验证集 0.62左右
#     4. 测试集 0.625 左右
#     
# ## 2018.10.17 第二次rank
# 0： CART-m1000-2018-10-17-10-55 : 
#     1. CART二叉回归树
#     2. 特征：增加特征1，并抛弃评论词汇数量
#     3. 超参数：max_node_size=1000,没有设置连续值特征搜索上限 ：
#     3. 验证集 0.605 左右
#     4. 测试集 61.3598
# 1: CART-m500-weight12018-10-17-11-23.csv
#     1. CART二叉回归树
#     2. 特征：增加特征1，保留原有特征
#     3. 超参数：
#         1. max_node_size=500,没有设置连续值特征搜索上限
#         2. 特征权重：
#             1. average_score,
#             2. Review_Total_Positive_Word_Counts
#             3. Review_Total_Positive_Word_Counts_radio_Total_Number_of_Reviews 在归一化后乘2
#     4. 验证集：0.627-0.645
#     5. 测试集：54.0939
# 2：CART-m500-weight1-no_feat2018-10-17-11-27.csv
#     1. CART二叉回归树
#     2. 特征：不修改原有特征
#     3. 超参数：
#         1. max_node_size=500,没有设置连续值特征搜索上限
#         2. 特征权重：
#             1. average_score,
#             2. Review_Total_Positive_Word_Counts
#             3. Review_Total_Positive_Word_Counts_radio_Total_Number_of_Reviews 在归一化后乘2
#     4. 验证集：0.632-0.640
#     5. 测试集：54.2822
# 3：CART-m500-no_weight-no_feat2018-10-17-11-33.csv
#     1. CART二叉回归树
#     2. 特征：不修改原有特征
#     3. 超参数：
#         1. max_node_size=500,没有设置连续值特征搜索上限
#         2. 特征权重：无
#     4. 验证集：0.623-0.632
#     5. 测试集：63.5761

# ## 2018.10.18 第二次rank
# 0：CART-m500-no_weight-add_tag_feat2018-10-18-23-54.csv
#     1. CART二叉回归树
#     2. 特征：增加tags提取出来的score特征
#     3. 超参数：
#         1. max_node_size = 500,没有设置连续值特征搜索上限
#         2. 特征权重：无
#     4. 验证集：
#     5. 测试集：63.4147
# 1：CART-m1000-no_weight-add_tag_feat2018-10-18-23-56.csv
#     1. CART二叉回归树
#     2. 特征：增加tags提取出来的score特征
#     3. 超参数：
#         1. max_node_size = 1000,没有设置连续值特征搜索上限
#         2. 特征权重：无
#     4. 验证集：0.626-0.641
#     5. 测试集：62.8266


