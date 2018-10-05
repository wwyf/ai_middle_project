
# coding: utf-8

# In[27]:


import sys
# sys.path.append(r'E:/0code')
# sys.path.append(r'/home/wangyf226/0code')
sys.path.append(r'/BIGDATA1/nsccgz_yfdu_1/asc19/wyf/pyml')


# In[28]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import pandas as pd
import numpy as np
from pyml.feature_extraction.text import CountVectorizer
from pyml.linear_model.classification import LogisticClassifier
from pyml.preprocessing import scale


# # 读取数据集

# 1. 读取训练数据和测试数据为字符串的列表
# 2. 读取训练集label，并转换为数字格式

# In[29]:


def read_train_text_to_list(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        contents = f.readlines()
    lines = [l.strip() for l in contents]
    return lines


# In[30]:


train_ori_X = read_train_text_to_list('../data/trainData.txt')
train_ori_Y = read_train_text_to_list('../data/trainLabel.txt')
train_ori_Y = np.array([int(y) for y in train_ori_Y])
test_ori_X = read_train_text_to_list('../data/testData.txt')


# # 数据预处理 & 特征工程
# 1. Count Vectors as feature
# 2. TF-IDF Vectors as festures
# 3. Word Embeddings as features
# 4. Text/NLP based features
# 5. Topic Models as features

# In[31]:


f_vier = CountVectorizer()
train_wc_X = f_vier.fit_transform(train_ori_X)
test_wc_X = f_vier.transform(test_ori_X)


# In[32]:


train_X = scale(train_wc_X)
test_X = scale(test_wc_X)


# In[33]:


# 训练模型


# In[ ]:


clf = LogisticClassifier()
clf.fit(train_X, train_ori_Y, watch=True)

