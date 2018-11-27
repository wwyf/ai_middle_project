import sys
# sys.path.append(r'E:/0code')
# sys.path.append(r'/home/wyf/0code')
# sys.path.append(r'/home/wangyf226/0code')
sys.path.append(r'/BIGDATA1/nsccgz_yfdu_1/asc19/wyf/0code')
import pandas as pd
import numpy as np
import datetime
import gensim
import re
import smart_open

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

vector_size = int(sys.argv[1])
print('vector_size : {}'.format(vector_size))
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
np.save('wyf-train_X-doc2vec-'+str(vector_size),train_X)
np.save('wyf-train_Y-doc2vec-'+str(vector_size),train_Y)
np.save('wyf-test_X-doc2vec-'+str(vector_size),test_X)
