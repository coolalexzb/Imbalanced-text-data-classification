#coding: utf-8


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
            features = [1 if word in text_words else 0 for word in feature_words]
        elif flag == 'svm':
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []
        ## -----------------------------------------------------------------------------------
        return features

    feature_list = [text_features(text, feature_words) for text in data_list]
    return feature_list