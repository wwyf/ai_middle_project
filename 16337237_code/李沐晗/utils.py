import os
import sys
import nltk
import gensim
import logging
import itertools
import multiprocessing
from bs4 import BeautifulSoup
import numpy as np
from typing import List, Any
from functools import partial

from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer

logger = logging.getLogger("utils_logger")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(asctime)s] : %(levelname)s : %(message)s "))
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

def is_path_accessible(pathname: str) -> bool:
    return pathname is not None and len(pathname) != 0 and os.access(pathname, os.R_OK)

def is_path_creatable(pathname: str) -> bool:
    """
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.
    """
    # Parent directory of the passed path. If empty, we substitute the current
    # working directory (CWD) instead.
    if pathname is None or len(pathname) == 0:
        return False
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)



def read_text(filename: str) -> List[str]:
    """
    @param filename filename
    @return list of strings of each line in the file
    """
    with open(filename, 'r', encoding='UTF-8') as file:
        contents = file.readlines()
    lines = [l.strip() for l in contents]
    return lines

def read_labels(filename: str) -> np.ndarray:
    with open(filename, 'r', encoding='UTF-8') as file:
        contents = file.readlines()
    labels = np.zeros(len(contents), dtype=np.int32)
    count = 0
    for line in contents:
        labels[count] = int(line)
        count += 1
    return labels

def save_labels(filename: str, labels: np.ndarray) -> None:
    with open(filename, 'w', encoding='UTF-8') as file:
        for l in labels:
            file.write("{}\n".format(l))

def read_array(filename: str) -> List[str]:
    return np.load(filename)

def save_array(filename: str, data: Any):
    np.save(filename, data)


"""             WORD2VEC            """
def _tokenize_w2v(paragraph):
    tk = TweetTokenizer()
    sno = nltk.stem.SnowballStemmer('english')
    # strip html tags
    soup = BeautifulSoup(paragraph, features="html.parser")
    paragraph = soup.get_text()

    # tokenize each sentence into seperated words(including emoticons)
    tokens = tk.tokenize(text=paragraph)
    new_tokens = [sno.stem(to.lower()) for to in tokens if len(to) > 1]
    return new_tokens

def tokenize_paragraph_w2v(paragraphs: List[str]) -> List[List[str]]:
    pool = multiprocessing.Pool()
    return pool.map(_tokenize_w2v, paragraphs)

def _compute_w2v(tokens, vector_size, model):
    count = 0
    tmp = np.zeros([len(tokens), vector_size], np.float32)
    for token in tokens:
        try:
            tmp[count] = model[token]
        except:
            pass
        count += 1
    return tmp

def compute_paragraph_word2vec(tokens: List[List[str]],
                               vector_size: int,
                               epochs=25,
                               workers=8,
                               min_count=10,
                               window=10,
                               model_path="",
                               load_model=False,
                               predict=False) -> Any:

    if not is_path_creatable(model_path) and not load_model:
        raise ValueError("Cannot save word2vec model to target path \"%s\"" % model_path)
    elif not is_path_accessible(model_path) and load_model:
        raise ValueError("Cannot read word2vec model from target path \"%s\"" % model_path)
    elif load_model:
        model = gensim.models.Word2Vec.load(model_path)
        vector_size = model.vector_size
        logger.info("word2vec model loaded")
    elif not load_model:
        model = gensim.models.word2vec.Word2Vec(tokens, size=vector_size, window=window,
                                                min_count=min_count, workers=workers)
        model.save(model_path)
        logger.info("word2vec model training finished")

    if not predict:
        return

    # compute vector for each paragraph or sentence
    pool = multiprocessing.Pool()
    result = pool.map(partial(_compute_w2v, vector_size=vector_size, model=model), tokens)
    return result

"""             DOC2VEC             """
def _tokenize_d2v(paragraph):
    tk = TweetTokenizer()
    # strip html tags
    soup = BeautifulSoup(paragraph, features="html.parser")
    paragraph = soup.get_text()
    # tokenize each paragraph into seperated senteces
    sentences = sent_tokenize(paragraph)
    # tokenize each sentence into seperated words(including emoticons)
    words = []
    for st in sentences:
        tokens = tk.tokenize(text=st)
        new_tokens = [to.lower() for to in tokens if len(to) > 1]
        words.append(new_tokens)
    final_words = []
    tmp = []
    for st in words:
        if len(tmp) > 30:
            final_words.append(tmp)
            tmp = st
        else:
            tmp += st
    final_words.append(tmp)
    return final_words

def tokenize_paragraph_d2v(paragraphs: List[str]) -> List[List[str]]:
    pool = multiprocessing.Pool()
    return pool.map(_tokenize_d2v, paragraphs)

def _compute_as_whole_d2v(paragraph, model):
    return model.infer_vector(list(itertools.chain.from_iterable(paragraph)))

def _compute_seperate_d2v(paragraph, vector_size, model):
    count = 0
    tmp = np.zeros([len(paragraph), vector_size], np.float32)
    for sentence in paragraph:
        tmp[count] = model.infer_vector(sentence)
        count += 1
    return tmp

def compute_paragraph_doc2vec(tokens: List[List[str]],
                              vector_size: int,
                              epochs=25,
                              workers=8,
                              min_count=10,
                              window=10,
                              model_path="",
                              load_model=False,
                              predict=False,
                              as_whole=False) -> Any:
    """
    take each sentence of the paragraph as input, apply doc2vec to each sentence seperatedly,
    or apply it to the whole paragraph
    @param tokens: tokens of the paragraph
    @return: doc2vec matrix, first dim = dim(doc2vec), second dim = number of sentences
             or: doc2vec vector, dim = dim(doc2vec)
    """
    if not is_path_creatable(model_path) and not load_model:
        raise ValueError("Cannot save doc2vec model to target path \"%s\"" % model_path)
    elif not is_path_accessible(model_path) and load_model:
        raise ValueError("Cannot read doc2vec model from target path \"%s\"" % model_path)
    elif load_model:
        model = gensim.models.Doc2Vec.load(model_path)
        vector_size = model.vector_size
        logger.info("doc2vec model loaded")
    elif not load_model:
        input = []
        if as_whole:
            count = 0 # IDs are used as labels
            for paragraph in tokens:
                input.append(gensim.models.doc2vec.TaggedDocument(list(itertools.chain.from_iterable(paragraph)),
                                                                  str(count)))
                count += 1
        else:
            count = 0
            for paragraph in tokens:
                for sentence in paragraph:
                    input.append(gensim.models.doc2vec.TaggedDocument(sentence, str(count)))
                    count += 1
        model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, window=window,
                                              min_count=min_count, workers=workers)
        model.build_vocab(input)
        model.train(documents=input,
                    epochs=epochs, total_examples=model.corpus_count, total_words=model.corpus_total_words)
        model.save(model_path)
        logger.info("doc2vec model training finished")

    if not predict:
        return

    # compute vector for each paragraph or sentence
    pool = multiprocessing.Pool()
    if as_whole:
        result = np.array(pool.map(partial(_compute_as_whole_d2v, model=model), tokens), dtype=np.float32)
        return result
    else:
        result = pool.map(partial(_compute_seperate_d2v, vector_size=vector_size, model=model), tokens)
        return result
