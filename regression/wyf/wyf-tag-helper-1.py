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
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk import pos_tag
from nltk.corpus import stopwords

# # 这一份文件的作用
#
# 这一份文件就是为了能够更好的提取tags的特征而弄的。
#
# 目前已经能够将tags数据，拆分成6个维度的数据，存在另一个excel表中

# # 读取数据文件

train = pd.read_excel('../data/train.xlsx')
test = pd.read_excel('../data/testStudent.xlsx')

train.dtypes # 检查有没有数据类型错误的，比如原本是int的变成str，说明里面可能有nan值等奇怪的数据

# train_ori_X = train.drop('Reviewer_Score', axis=1)
train_ori_X = train
train_ori_Y = train['Reviewer_Score']
test_ori_X = test

# # 特征工程

train_tags = train['Tags']

def tags_tokenize(s):
    ss = s.split(',')
    return [ i.strip(",'[] '").strip(",'[ ]'") for i in ss]

def tags_to_words(df):
    tags_df = df['Tags']
    words = set()
    for tag in tags_df:
        for word in tags_tokenize(tag):
            words.add(word)
    return words

all_words = tags_to_words(train_ori_X)
all_words |= tags_to_words(test_ori_X)

trip_words = set()
traveler_words = set()
room_types = set()
order_types = set()
night_types = set()

# 提取旅游类型
for word in all_words:
    if 'trip' in word:
        trip_words.add(word)
print(trip_words,len(trip_words))
# 去除已提取旅游类型
all_words = [word for word in all_words if word not in trip_words]

len(all_words)

# 提取旅游者类型
for word in all_words:
    if 'traveler' in word:
        traveler_words.add(word)
#     elif 'Adults' in word:
#         traveler_words.add(word)
traveler_words.add('Couple')
traveler_words.add('Group')
traveler_words.add('Family with young children')
traveler_words.add('Family with older children')
traveler_words.add('Travelers with friends')
print(traveler_words, len(traveler_words))
# 去除已提取
all_words = [word for word in all_words if word not in traveler_words]

len(all_words)

# 提取住了多少晚
for word in all_words:
    if 'Stayed' in word and 'night' in word:
        night_types.add(word)
print(night_types,len(night_types))
# 去除已提取
all_words = [word for word in all_words if word not in night_types]

len(all_words)

# 提取订单来源
for word in all_words:
    if 'Submitted from' in word:
        order_types.add(word)
print(order_types,len(order_types))
# 去除已提取
all_words = [word for word in all_words if word not in order_types]

len(all_words)

# 提取定的房间类型
for word in all_words:
    if 'Room' in word:
        room_types.add(word)
    elif 'room' in word:
        room_types.add(word)
print(room_types,len(room_types))
# 去除已提取
all_words = [word for word in all_words if word not in room_types]

len(all_words)

def get_type(ss, types):
    tags = tags_tokenize(ss)
    for s in tags:
        # 判断旅游类型
        if s in types:
            return types.index(s)
    return -1

# ## 增加旅游类型特征

trip_type_lists = list(trip_words)
trip_type_lists

def get_trip_type_feature(df):
    """
    处理数据集中的tags,判断其旅游类型
    """
    df = df.copy()
    df['TripType'] = df.apply(lambda s : get_type(s['Tags'], trip_type_lists),axis=1)
    return df

# ## 增加旅游者类型特征

traveler_type_list = list(traveler_words)
traveler_type_list

def get_traveler_type_feature(df):
    """
    判断数据集中的旅游类型
    """
    df = df.copy()
    df['traveler_type'] = df.apply(lambda s : get_type(s['Tags'], traveler_type_list), axis=1)
    return df

# ## 增加订单类型特征

order_type_list = list(order_types)
order_type_list

# 增加是否从移动设备上提交订单
def get_order_type_feature(df):
    """
    判断数据集中的订单提交类型
    """
    df = df.copy()
    df['order_type'] = df.apply(lambda s : get_type(s['Tags'], order_type_list)+1, axis=1)
    return df

# ## 增加居住天数特征

import re

# 增加居住天数
def get_nights(ss, types):
    pattern = re.compile(r'[a-zA-Z]+\s(\d+)\s[a-zA-Z]+')
    tags = tags_tokenize(ss)
    for s in tags:
        # 判断旅游类型
        if s in types:
            m = pattern.match(s)
            return int(m.group(1))
    return -1

# 增加居住天数
def get_night_type_feature(df):
    """
    判断数据集中的订单提交类型
    """
    df = df.copy()
    df['nights_num'] = df.apply(lambda s : get_nights(s['Tags'], night_types), axis=1)
    return df

# ## 增加宠物特征，如果有宠物则为1，没有就为0

pet_lists = ['With a pet']

def get_pet(ss):
    tags = tags_tokenize(ss)
    for s in tags:
        if s == 'With a pet':
            return 1
    return 0

def get_pet_feature(df):
    """
    增加宠物特征
    """
    df.copy()
    df['with_pet'] = df.apply(lambda s : get_pet(s['Tags']), axis=1)
    return df

# ## 将剩余无用特征放回去

# 将剩余没用到的特征放回到表格中
def get_no_used_tags(ss):
    tags = tags_tokenize(ss)
    no_used_features = []
    for s in tags:
        if s not in trip_words and s not in traveler_words and s not in order_types and s not in night_types and s not in pet_lists:
            no_used_features.append(s)
    if len(no_used_features) > 1:
        print(no_used_features)
    return no_used_features
def get_room_type_tags(ss):
    tags = tags_tokenize(ss)
    for s in tags:
        if s not in trip_words and s not in traveler_words and s not in order_types and s not in night_types and s not in pet_lists:
            return s
    return -1

# 增加无用特征
def get_no_used_tag_feature(df):
    """
    判断数据集中的订单提交类型
    """
    df = df.copy()
    df['no_used'] = df.apply(lambda s : get_no_used_tags(s['Tags']), axis=1)
    return df
def get_room_type_feature(df):
    """
    判断数据集中的订单提交类型
    """
    df = df.copy()
    df['room_type'] = df.apply(lambda s : get_room_type_tags(s['Tags']), axis=1)
    return df

# 增加旅游类型一列，如果没有就设置为-1
train_X_feat = get_trip_type_feature(train_ori_X)
test_X_feat = get_trip_type_feature(test_ori_X)

train_X_feat = get_traveler_type_feature(train_X_feat)
test_X_feat = get_traveler_type_feature(test_X_feat)

train_X_feat = get_order_type_feature(train_X_feat)
test_X_feat = get_order_type_feature(test_X_feat)

train_X_feat = get_night_type_feature(train_X_feat)
test_X_feat = get_night_type_feature(test_X_feat)

train_X_feat = get_pet_feature(train_X_feat)
test_X_feat = get_pet_feature(test_X_feat)

# +
# train_X_feat = get_no_used_tag_feature(train_X_feat)
# test_X_feat = get_no_used_tag_feature(test_X_feat)
# -

train_X_feat = get_room_type_feature(train_X_feat)
test_X_feat = get_room_type_feature(test_X_feat)

train_X_feat.head()

train_X_feat.drop(labels='Tags', axis=1)
test_X_feat.drop(labels='Tags', axis=1)

train_X_feat.to_excel('../data/train_add_feat.xlsx', index_label=None, index=0)
test_X_feat.to_excel('../data/test_add_feat.xlsx', index_label=None, index=0)
