#coding: utf-8
from gensim.models import word2vec
from functools import reduce
import time
# 提取新闻特征，即事件抽取
def TextFeatures(data_list, feature_words, flag='sklearn'):
    def text_features(text, feature_words):
        text_words = set(text)
        ## -----------------------------------------------------------------------------------
        if flag == 'nltk':
            ## nltk特征 dict
            features = {word: 1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            ## sklearn特征 list
            features = [1 if len(set(word) & text_words) != 0 else 0 for word in feature_words]
        elif flag == 'svm':
            # features_tmp = [list(model[word[0]]) if len(set(word) & text_words) != 0 else empty for word in feature_words]
            # features = reduce(lambda x, y: x + y, features_tmp)
            features = [1 if len(set(word) & text_words) != 0 else 0 for word in feature_words]
        else:
            features = []
        ## -----------------------------------------------------------------------------------
        return features

    #empty = [0] * 10
    #model = word2vec.Word2Vec.load("./word2vec_wx.model")
    feature_list = [text_features(text, feature_words) for text in data_list]
    return feature_list