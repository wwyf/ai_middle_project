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
import gensim
import re
import smart_open
from pyml.feature_extraction.text import CountVectorizer
from pyml.linear_model.classification import LogisticClassifier
from pyml.neighbors.classification import KNeighborsClassifier
from pyml.metrics.classification import precision_score
from pyml.model_selection import KFold
from pyml.model_selection import ShuffleSplit
from pyml.preprocessing import StandardScaler

# # 读取数据
# 1. 读取训练数据和测试数据为字符串的列表
# 2. 读取训练集label，并转换为数字格式

def read_train_text_to_list(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        contents = f.readlines()
    lines = [l.strip() for l in contents]
    return lines

# train_ori_X = read_train_text_to_list('../data/trainData.txt')
train_ori_Y = read_train_text_to_list('../data/trainLabel.txt')
train_ori_Y = np.array([int(y) for y in train_ori_Y])
# test_ori_X = read_train_text_to_list('../data/testData.txt')

def read_raw_documents(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="UTF-8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                # 变小写，去标点符号，分词
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

train_sentences = list(read_raw_documents('../data/trainData.txt'))
test_sentences = list(read_raw_documents('../data/testData.txt', tokens_only=True))

# # 数据预处理 & 特征工程
# 1. Count Vectors as feature
# 2. TF-IDF Vectors as festures
# 3. Word Embeddings as features
# 4. Text/NLP based features
# 5. Topic Models as features

vector_size = 100
model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=40)

model.build_vocab(train_sentences)

model.train(train_sentences, total_examples=model.corpus_count, epochs=model.epochs)

n_train_samples = len(train_sentences)
n_test_samples = len(test_sentences)
train_X = np.zeros((n_train_samples, vector_size))
test_X = np.zeros((n_test_samples, vector_size))
for i in range(0, n_train_samples):
    train_X[i] = model.infer_vector(train_sentences[i][0])
for i in range(0, n_test_samples):
    test_X[i] = model.infer_vector(test_sentences[i])

train_Y = train_ori_Y

train_X.shape,train_Y.shape

# # 交叉验证

k_range = range(2,21)
n_splits = 2
ms = ShuffleSplit(n_splits=n_splits)
k_scores = np.zeros((len(k_range)))
for train_indices, test_indices in ms.split(train_X):
    for i,k in enumerate(k_range):
        clf = KNeighborsClassifier(k=k)
        clf.fit(train_X[train_indices], train_Y[train_indices])
        y_pred = clf.predict(train_X[test_indices])
        score = precision_score(train_Y[test_indices], y_pred)
        print('k : {} score: {}'.format(k, score))
        k_scores[i] += score
avg_k_scores = k_scores/n_splits
print(avg_k_scores)
print("best k ", np.argmax(avg_k_scores)+1)

for k in [13,14,15,18,19,20]:
    clf = KNeighborsClassifier(k=k)
    clf.fit(train_X,train_Y)
    y_pred = clf.predict(test_X)
    sub = pd.DataFrame(y_pred)
    sub.to_csv('../results/'+'KNN-'+str(k)+'-'+ str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + ".csv", index=0, header=None, index_label=None)


