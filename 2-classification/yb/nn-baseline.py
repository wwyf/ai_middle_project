import sys
# sys.path.append(r'E:/0code')
#sys.path.append('/Users/yanbin/Documents/Projects/AI-Middle-Project/')
#sys.path.append('/Users/yanbin/Documents/Projects/mylearn')

sys.path.append('/home/wyf/0code/AI-Middle-Project/')
sys.path.append('/home/wyf/0code/mylearn')
import numpy as np
import smart_open
import gensim
from logger import get_logger

mylogger = get_logger(__name__)
mylogger.debug('hello world')

def read_train_text_to_list(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        contents = f.readlines()
    lines = [l.strip() for l in contents]
    return lines

def read_raw_documents(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="UTF-8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                # 变小写，去标点符号，分词
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

def load_dataset():
    ds = np.DataSource()
    if ds.exists('trainX.npy') and ds.exists('trainY.npy'):
        mylogger.info('exist saved file. load.')
        trainX = np.load('trainX.npy')
        trainY = np.load('trainY.npy')
        return trainX, trainY
    # train_ori_X = read_train_text_to_list('../data/trainData.txt')
    train_ori_Y = read_train_text_to_list('../data/trainLabel.txt')
    train_ori_Y = np.array([int(y) for y in train_ori_Y])
    # test_ori_X = read_train_text_to_list('../data/testData.txt')

    train_sentences = list(read_raw_documents('../data/trainData.txt'))
    test_sentences = list(read_raw_documents('../data/testData.txt', tokens_only=True))

    # # 数据预处理 & 特征工程
    # 1. Count Vectors as feature
    # 2. TF-IDF Vectors as festures
    # 3. Word Embeddings as features
    # 4. Text/NLP based features
    # 5. Topic Models as features

    vector_size = 50
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=40)

    model.build_vocab(train_sentences)

    model.train(train_sentences, total_examples=model.corpus_count, epochs=model.epochs)

    n_train_samples = len(train_sentences)
    n_test_samples = len(test_sentences)
    vector_size = 50
    train_X = np.zeros((n_train_samples, vector_size))
    test_X = np.zeros((n_test_samples, vector_size))
    for i in range(0, n_train_samples):
        train_X[i] = model.infer_vector(train_sentences[i][0])
    for i in range(0, n_test_samples):
        test_X[i] = model.infer_vector(test_sentences[i])

    train_Y = train_ori_Y

    train_X.shape
    np.save('trainX', train_X)
    np.save('trainY', train_Y)
    return train_X, train_Y
    
train_X, train_Y = load_dataset()
print(train_X, train_Y)
