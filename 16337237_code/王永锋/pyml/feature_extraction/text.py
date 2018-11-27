import re
import numpy as np
from orderedset import OrderedSet

def word_tokenizer(sentence):
    """
    将一句话分成单词的列表, 会顺便承担清除标点符号的工作

    Parameters
    ----------
    sentence : string

    Returns
    --------
    words : the list of word
    """
    solve_sentence = re.sub(r'\W+|\s+', ' ', sentence)
    words = solve_sentence.lower().strip(' ').split(' ')
    return words

class CountVectorizer():
    """
    Parameters
    -----------


    """
    def __init__(self):
        # 这一个表是用来查同义词的，可以将一些具有相同意义的字符串转换成一样的字符串
        self.lookup_table = {}
        # TODO: 待完善
        self.comfused_words_set = set()
        self.not_comfused_words_set = set()
        self.word_ordered_set = OrderedSet()
        self.dictionary = {}
        self.dictionary_len = len(self.dictionary)
        pass



    
    def fit(self, raw_documents):
        """

        build(or add) the dictionary in class

        Parameters
        ------------
        raw_documents : iterable

        """
        for text in raw_documents:
            words = word_tokenizer(text)
            for word in words:
                self.word_ordered_set.add(word)
        for word in self.word_ordered_set:
            self.dictionary[word] = self.word_ordered_set.index(word)
        self.dictionary_len = len(self.dictionary)
    

    def transform(self, raw_documents):
        """
        Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary fitted with fit or the one provided to the constructor.

        Parameters
        ------------
        raw_documents : iterable

        Returns
        --------
        X : 2d array-like sparse matrix, shape(n_samples, n_features)
        """
        text_matrix = np.zeros((len(raw_documents), self.dictionary_len+1))
        for i, document in enumerate(raw_documents):
            words = word_tokenizer(document)
            for word in words:
                if word in self.dictionary:
                    text_matrix[i,self.dictionary[word]] += 1
                else :
                    text_matrix[i,self.dictionary_len] = 1
        return text_matrix
    
    def fit_transform(self, raw_documents):
        """
        Transform documents to document-term matrix.
        This is equivalent to fit followed by transform, but more efficiently implemented.
        Parameters
        ------------
        raw_documents : iterable

        Returns
        --------
        X : 2d array-like sparse matrix, shape(n_samples, n_features)
        """
        self.fit(raw_documents)
        return self.transform(raw_documents)

class TfidfVectorizer():
    def __init__(self):
        self.word_ordered_set = OrderedSet()
        self.dictionary = {}
        self.dictionary_len = len(self.dictionary)
        pass
    
    def fit(self, raw_documents):
        """

        build(or add) the dictionary in class

        Parameters
        ------------
        raw_documents : iterable

        """
        for text in raw_documents:
            words = word_tokenizer(text)
            for word in words:
                self.word_ordered_set.add(word)
        for word in self.word_ordered_set:
            self.dictionary[word] = self.word_ordered_set.index(word)
        self.dictionary_len = len(self.dictionary)
    

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def transform(self, raw_documents):
    # def get_TD_IDE_mat(raw_documents):
        """
        Parameters
        ------------
        raw_documents : iterable of string

        Returns
        -----------
        df_idf_mat : TD_IDE矩阵，行数为文段数量，列数为不重复单词数量
        """
        # 计算矩阵维度
        row_d = len(raw_documents)
        column_d = self.dictionary_len+1
        # print(row_d, column_d)
        # 初始化矩阵
        num_mat = np.zeros((row_d, column_d))
        df_mat = np.zeros((row_d, column_d))
        idf_mat = np.zeros((1, column_d))
        # 遍历每一行，计算num_mat, num_mat[i][j]表示第i行句子中单词j出现的个数
        for row_index, row in enumerate(raw_documents):
            words = word_tokenizer(row)
            for word in words:
                if word in self.dictionary:
                    num_mat[row_index][[self.dictionary[word]]] += 1
                else :
                    num_mat[row_index][[self.dictionary_len]] = 1
        # 计算df矩阵
        df_mat = num_mat/num_mat.sum(axis=1).reshape(row_d, 1)
        # 计算idf值 
        count_mat = num_mat
        count_mat[count_mat != 0] = 1
        idf_mat = np.log(row_d/(count_mat.sum(axis=0)+1).reshape(1, column_d))
        # 得到了df_mat, idf_mat
        df_idf_mat = df_mat * idf_mat
        return df_idf_mat



    
    # def word_convert(sentence):
    #     """
    #     使用look up表，将句子中一些可能常见的词根据look up表替换
    #     如 "'s" -> " is "
    #     """
    #     for confused_word,not_confused_word in self.lookup_table.items():
    #         sentence = sentence.replace(confused_word, not_confused_word)
    #     return sentence

    # def add_comfused_word_pair(comfused_word, not_comfused_word):
    #     """
    #     Parameters
    #     -----------
    #     comfused_word : str
    #     not_comfused_word : str

    #     Returns
    #     ---------
    #     None
    #     """
    #     self.lookup_table[comfused_word] = not_comfused_word


if __name__ == '__main__':
    test_string =r"""** CONTAINS SPOILERS ** <br /><br />The truly exquisite Sean Young (who in some scenes, with her hair poofed up, looks something like Elizabeth Taylor) is striking in her opening moments in this film. Sitting in the back of a police car waiting to signal a bust, her face and body are tense and distracted. Unfortunately, once the bust is over Young's strained demeanor never changes. This is one fatally inhibited actress.<br /><br />One has only to compare Young to the performer playing her coworker and best friend, Arnetia Walker, to grasp what is missing in Young. Walker is open, emotional, and at ease at all times...in that there's no apparent barrier between what she may be feeling and her expression of it. She is an open book. Young, on the other hand, acts in the skittish, self-conscious way you might expect your neighbor to act were they suddenly thrown into starring in a film. Basically, she doesn't have a clue.<br /><br />With this major void looming at the center of the movie, we're left to ponder the implausiblities of the story. For instance, after Miss Young is kidnapped by the criminal she's trailing and locked in a closet, she breaks the door down when left alone. Granted, she's dressed only in a bra and panties, but in a similar situation, with a psycho captor due to return any moment, would you head for the door...or take the time to go through his dresser, take out some clothes and get dressed? I would guess that this and other scenes are trying to suggest some sort of mixed emotions Miss Young's character is experiencing, but Young can not convey this type of complexity.<br /><br />There are a few affecting moments in the film, such as the short police interviews with the criminal's past victims, but overall this is an aimless endeavor. It's too bad Miss Young was replaced while filming the pair of comic book style films that might have exploited her limitations with some humor (BATMAN and DICK TRACY), because her floundering while attempting to play actual people is oddly touching. Watching Miss Young try to act, at least in this "thriller", is a sad spectacle. '"""
    test_string2 = r"""I love ghost stories and I will sit through a movie til it's end, even if I'm not really enjoying it. I rarely feel like I wasted my time... BUT, this adaptation of the Bell Witch story was horrible! <br /><br />It wasn't scary in the least bit. What is with the comic relief moments? The dialog was tedious. Acting inconsistent The movie was WAY too long and some scenes were unnecessarily drawn out in my open. (Like the birthday party)<br /><br />The only good think I can think about mentioning is the costumes and props were well done.<br /><br />I am curious about other adaptation, but until then, I will stick to reading about the story."""
    raw_documents = [test_string, test_string2]
    sentences = ['europe retain trophy with big win', 'senate votes to revoke pensions']
    # cv = CountVectorizer()
    # print(cv.fit_transform(raw_documents))
    tv = TfidfVectorizer()
    print(tv.fit_transform(raw_documents))