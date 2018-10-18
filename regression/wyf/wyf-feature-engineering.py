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

import pandas as pd
import pandas as pd
import numpy as np
import datetime

train = pd.read_excel('../data/train_add_feat.xlsx')
test = pd.read_excel('../data/test_add_feat.xlsx')

train.head()

t_g = train.groupby('room_type')

t_g.mean().sort_values(by='Reviewer_Score', ascending=False )

# ## 特征数字化

# 思路是：
# 先对某离散特征进行聚合，计算平均分，然后根据平均分进行排序，按升序给特征进行编号。

feature_name = 'traveler_type'
type_group = train.groupby(feature_name)

ascending_df = type_group.mean().sort_values(by='Reviewer_Score').reset_index().loc[:,[feature_name, 'Reviewer_Score']]
ascending_df = ascending_df.drop(labels='Reviewer_Score',axis=1)
result = ascending_df.reset_index()
result.columns = [feature_name+'_score', feature_name ]

result

pd.merge(train, result, on=feature_name, how='left')

def get_discrete_to_score_feature(df, df_test, feature_name):
    type_group = df.groupby(feature_name)
    ascending_df = type_group.mean().sort_values(by='Reviewer_Score').reset_index().loc[:,[feature_name, 'Reviewer_Score']]
    ascending_df = ascending_df.drop(labels='Reviewer_Score',axis=1)
    result = ascending_df.reset_index()
#     print(result)
    result.columns = [feature_name+'_score', feature_name ]
    return pd.merge(df, result, on=feature_name, how='left'),pd.merge(df_test, result, on=feature_name, how='left')

get_discrete_to_score_feature(train,test, 'traveler_type' )[1]

get_discrete_to_score_feature(train, 'room_type' )

# ## 真正特征数字化

feature_names = ['TripType','traveler_type','order_type','nights_num','with_pet','room_type']

train = pd.read_excel('../data/train_add_feat.xlsx')
test = pd.read_excel('../data/test_add_feat.xlsx')
train_feat = train
test_feat = test

for feature_name in feature_names:
    train_feat,test_feat = get_discrete_to_score_feature(train_feat,test_feat,feature_name)
    train_feat = train_feat.drop(labels=feature_name,axis=1)
    test_feat = test_feat.drop(labels=feature_name,axis=1)

train_feat.columns

train_feat = train_feat.drop(labels='Tags', axis=1)
test_feat = test_feat.drop(labels='Tags', axis=1)

train_feat.columns

train_feat.to_excel('../data/train_add_feat_score.xlsx', index=0, index_label=None)
test_feat.to_excel('../data/test_add_feat_score.xlsx', index=0, index_label=None)


