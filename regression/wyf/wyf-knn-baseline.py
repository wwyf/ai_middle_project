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
#   toc:
#     base_numbering: 1
#     nav_menu: {}
#     number_sections: true
#     sideBar: true
#     skip_h1_title: false
#     title_cell: Table of Contents
#     title_sidebar: Contents
#     toc_cell: false
#     toc_position: {}
#     toc_section_display: true
#     toc_window_display: false
# ---

import sys
# sys.path.append(r'E:/0code')
sys.path.append(r'/home/wyf/0code')
# sys.path.append(r'/home/wangyf226/0code')
# sys.path.append(r'/BIGDATA1/nsccgz_yfdu_1/asc19/wyf/pyml')

import pyml
import pandas as pd
import numpy as np
from pyml.neighbors.regression import KNeighborsRegressor
from pyml.preprocessing import z_score

# # 读取数据文件

train = pd.read_excel('./data/train.xlsx')
test = pd.read_excel('./data/testStudent.xlsx')

train.dtypes # 检查有没有数据类型错误的，比如原本是int的变成str，说明里面可能有nan值等奇怪的数据

train.Tags[0].strip("[]").replace("'",'').split(',')

def preprocessing_data(df):
    pass

# # 构建训练集与测试集

train_label = train['Reviewer_Score']
train_feat = train.drop('Reviewer_Score', axis=1)
test_feat = test

# # 定义特征构建函数
#
# https://github.com/ShawnyXiao/2017-CCF-BDCI-Enterprise 该师兄的分析我觉得蛮有道理的
#
#
# 1. 基础特征
# 2. 偏离值特征
# 3. 交叉特征
# 4. 想象力特征hhh

# 观察数据，发现
# 1. 通过比值构建新特征：构造积极、消极、评论员评论占比特征

def get_proportion_feature(df):
    df = df.copy()
    
    base_features = ['Total_Number_of_Reviews']
    gap_features = ['Review_Total_Negative_Word_Counts', 'Review_Total_Positive_Word_Counts','Total_Number_of_Reviews_Reviewer_Has_Given']
    for base_feature in base_features:
        for gap_feature in gap_features:
            df[gap_feature+'_radio_'+base_feature] = df[gap_feature]/df[base_feature] * 10
            # 数字太小了，乘上一个10
            df = df.drop(gap_feature, axis=1)
    return df

def normalized_df(df):
    # 将数据归一化
    return (df - df.min()) / (df.max() - df.min())

# # 调用函数 构建特征

# 暂时不处理tags数据，就将其丢弃
train_feat = train_feat.drop(labels=['Tags'], axis=1)
test_feat = test_feat.drop(labels=['Tags'], axis=1)

train_feat = get_proportion_feature(train_feat)
test_feat = get_proportion_feature(test_feat)

train_feat = normalized_df(train_feat)
test_feat = normalized_df(test_feat)

# # 导入模型与训练

kreg = KNeighborsRegressor()

kreg.fit(train_feat.values, train_label.values.reshape(-1,1))

y_pred = kreg.predict(test_feat.values, watch=True)



# # 结果文件的写出

sub = pd.DataFrame(y_pred)
sub.to_csv('./results/'+ str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + ".csv", index=0, header=None, index_label=None)


