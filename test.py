# coding: utf-8

import os
import jieba
import nltk
from sklearn.naive_bayes import MultinomialNB
import jieba.analyse
from sklearn.svm import SVC
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from SFeature import *
import re

import time
import random
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
#import MySQLdb
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest ,chi2
from sklearn.decomposition import NMF, LatentDirichletAllocation

# 生成停用词字典
def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r',encoding='utf-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word) > 0 and word not in words_set:  # 去重
                words_set.add(word)
    return words_set
'''
    数据预处理，原函数因为样本数量有限，对样本做了训练集和测试集的划分，此处不需要划分，
    可以详细看数据处理流程，不懂的地方可用print输出看结果
'''
def TextProcessingsep(test_path):

    test_data_list = []
    test_class_list = []

    folder_list = os.listdir(test_path)
    for folder in folder_list:
        new_folder_path = os.path.join(test_path, folder)
        files = os.listdir(new_folder_path)
        # 类内循环
        j = 1
        for file in files:
            if j > 500:  # 每类text样本数最多100
                break
            with open(os.path.join(new_folder_path, file).replace('\\', '/'), 'r',encoding='utf-8') as fp:
                raw = fp.read()
            ## jieba分词
            # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
            word_cut = jieba.cut(raw, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
            word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
            # jieba.disable_parallel() # 关闭并行分词模式
            test_data_list.append(word_list)
            test_class_list.append(folder)
            j += 1

    return test_data_list, test_class_list


# 提取特征词
def words_dict(all_words_list, deleteN, stopwords_set=set()):
    # 选取特征词
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:  # feature_words的维度1000
            break
        # print all_words_list[t]
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
            n += 1
    for i in feature_words:
        print ("fea:     " + i)
    print ("--------------finish--------------")
    return feature_words

# 从每类新闻训练集中抽取指定数量的feature words
def words_dict_new(all_words_list,stopwords_set=set()):
    # 统计词频放入all_words_dict
    feature_words = []
    n = 1
    for t in range(0, len(all_words_list), 1):
        if n > 2000:  # feature_words的维度1000
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                    all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words

def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag='nltk'):
    test_accuracy =[]
    if flag == 'nltk':
        ## nltk分类器
        train_flist = zip(train_feature_list, train_class_list)
        test_flist = zip(test_feature_list, test_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        # print classifier.classify_many(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.classify(test_feature),
        test_accuracy = nltk.classify.accuracy(classifier, test_flist)
    elif flag == 'sklearn':
        ## sklearn分类器
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        print ("pre: ")
        x = classifier.predict(test_feature_list)
        # print test_feature_list
        print (classifier.predict(test_feature_list))
        # y=x.tolist()
        # print y[0]
        # for test_feature in test_feature_list:
        #     print classifier.predict(test_feature)[0]
        test_accuracy = classifier.score(test_feature_list, test_class_list)
        print (test_accuracy)
    elif flag == 'svm':
        classifier = SVC(kernel='linear')
        classifier.fit(train_feature_list, train_class_list)
        print ("presvm:")
        r =  classifier.predict(test_feature_list)
        print (r)
        n = 0
        for i in list(r) :
            n +=1
            if n%100 ==0:
                res.write('\n')
            res.write(str(i) + " ")
        res.close()
        test_accuracy = classifier.score(test_feature_list, test_class_list)
        print (test_accuracy)
    elif flag == 'GussNB':
        classifier = GaussianNB().fit(train_feature_list, train_class_list)
        print ("pre: ")
        print (classifier.predict(test_feature_list))
    elif flag == 'lda':
        lda = LinearDiscriminantAnalysis(solver="svd",store_covariance=True)
        lda_res = lda.fit(train_feature_list,train_class_list)
        print (lda_res.predict(test_feature_list))
        test_accuracy = lda.score(test_feature_list,test_class_list)
        print (test_accuracy)
    else:
        test_accuracy = []
    return test_accuracy


def TextClassifieracc(train_feature_list, test_feature_list, train_class_list, flag='nltk'):
    if flag == 'nltk':
        ## nltk分类器
        train_flist = zip(train_feature_list, train_class_list)
        # test_flist = zip(test_feature_list, test_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        # print classifier.classify_many(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.classify(test_feature),
        # print ''
        # test_accuracy = nltk.classify.accuracy(classifier, test_flist)
    if flag == 'sklearn':
        ## sklearn分类器
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        # print "pre: "
        # print classifier.predict(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.predict(test_feature)[0]
        test_accuracy = classifier.score(test_feature_list, test_class_list)
    else:
        test_accuracy = []
    print (test_accuracy)
    return test_accuracy



def dimension_reduce(train_data_list_f, test_data_list_f):
    corpus_train = []
    corpus_test = []
    train_feature_list = []
    test_feature_list = []
    for text in train_data_list_f:
        line_feature = jieba.analyse.textrank("".join(text), topK=10)
        line = "/".join(line_feature)
        corpus_train.append(line)
    for text in test_data_list_f:
        line_feature = jieba.analyse.textrank("".join(text), topK=10)
        line = "/".join(line_feature)
        corpus_test.append(line)
    # newsgroup_train.data is the original documents, but we need to extract the
    # feature vectors inorder to model the text data

    '''
    vectorizer = HashingVectorizer(non_negative=True,n_features=10000)
    fea_train = vectorizer.fit_transform(corpus_train)
    fea_test = vectorizer.fit_transform(corpus_test)
    '''
    tv = TfidfVectorizer(sublinear_tf=True, max_df=0.5, )
    tfidf_train_2 = tv.fit_transform(corpus_train)
    tv2 = TfidfVectorizer(vocabulary=tv.vocabulary_)
    tfidf_test_2 = tv2.fit_transform(corpus_test)
    analyze = tv.build_analyzer()
    tv.get_feature_names()  # statistical features/terms
    '''
    tfidfvec = TfidfVectorizer()
    train_cop_tfidf = tfidfvec.fit_transform(corpus_train)
    weight_train = train_cop_tfidf.toarray()
    '''
    '''
    for i in weight_train:
        train_sort=sorted(list(i))
        feature=[]
        for j in train_sort:
            if j!=0 and j<8:
                feature.append(j)
        train_feature_list.append(feature)
    '''
    #pca_train = PCA(n_components=50)
    #train_feature_list = pca_train.fit_transform(tfidf_train_2)

    '''
    for text in test_data_list_f:
        line_feature = jieba.analyse.textrank("".join(text), topK=10)
        line = "/".join(line_feature)
        corpus_test.append(line)
    tfidfvec = TfidfVectorizer()
    test_cop_tfidf = tfidfvec.fit_transform(corpus_test)
    weight_test = test_cop_tfidf.toarray()
    '''

    '''
    for i in weight_test:
        test_sort=sorted(list(i))
        feature=[]
        for j in test_sort:
            if j!=0 and j<8:
                feature.append(j)
        test_feature_list.append(feature)
    '''

    # pca_test = PCA(n_components=50)
    # test_feature_list = pca_test.fit_transform(tfidf_test_2)

    '''
    train_nmf = NMF(n_components=10, random_state=1,
              beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
              l1_ratio=.5).fit(train_cop_tfidf)
    test_nmf = NMF(n_components=10, random_state=1,
                    beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
                    l1_ratio=.5).fit(test_cop_tfidf)
    '''

    # print(type(train_cop_tfidf))
    # print(type(train_feature_list))
    # print(type(test_feature_list))
    return tfidf_train_2, tfidf_test_2

    # return train_nmf, test_nmf
    # return train_feature_list, test_feature_list

def remove_stopwords(test_data_list, stopwords_set):
    test_data_filter = []
    for i in test_data_list:
        filter = []
        for j in i:
            if j not in stopwords_file:
                filter.append(j)
        test_data_filter.append(filter)
    return test_data_filter

#结果存入数据库
def store_database():
    pass
    # c=[count]*num
    # print (count,allpre)
    # allpre=list(map(lambda x: int(x[0]) / int(x[1]), zip(allpre, c)))
    # all=[]
    # print (allpre)
    # d={'1':"财经",'2':"科技",'3':"汽车",'4':"房产",'5':"体育",'6':"娱乐",'7':"其他"}
    # for i in allpre:
    #     i=str(i)
    #     all.append(d[i])
    # # print all
    # conn = MySQLdb.connect(
    #     host='localhost',
    #     port=3306,
    #     user='root',
    #     passwd='root',
    #     db='crawler',
    # )
    # cur = conn.cursor()
    # conn.set_character_set('utf8')
    # cur.execute('SET NAMES utf8;')
    # cur.execute('SET CHARACTER SET utf8;')
    # cur.execute('SET character_set_connection=utf8;')
    # tmp=zip(news_id,all,news_id)
    # for i in tmp:
    #         #print i
    #     try:
    #         sqli = "insert into " + "crawler_sina_classifier " + "values(%s,%s,%s)"
    #         # print "_________________________"
    #         cur.execute(sqli,i)
    #         print ("================================")
    #     except Exception,e:
    #         print (e)
    #         c=0
    #
    # cur.close
    # conn.commit()
    # conn.close()

if __name__ == '__main__':

    print("start")

    train_class_list = []
    feature_words = []
    train_feature_list = []
    file_trainclass = open("train_class.txt", 'r', encoding='UTF-8')
    file_features = open("feature_words.txt", 'r', encoding='UTF-8')
    file_traindata = open("matrix_fea.txt", 'r', encoding='UTF-8')
    res = open("res.txt", 'w', encoding="utf-8")

    for lines in file_trainclass:
        train_class_list.append(lines.strip())
    for lines in file_features:
        feature_words.append(lines.strip().split(" "))
    for lines in file_traindata:
        tmp = lines.strip().split(',')
        l = [float(i) for i in tmp]
        train_feature_list.append(l)

    test_path = "./newdata/test/"
    test_data_list, test_class_list = TextProcessingsep(test_path)

    # 生成stopwords_set
    stopwords_file ="./stopwords_cn.txt"
    stopwords_set = MakeWordsSet(stopwords_file)
    ## 文本特征提取和分类
    num = 100
    flag = 'svm'
    test_accuracy_list = []
    #allpre = [0] * num

    test_data_list_f = remove_stopwords(test_data_list, stopwords_set)
    test_feature_list = TextFeatures(test_data_list_f, feature_words, flag)


    #调用降维算法
    #train_feature_list, test_feature_list = dimension_reduce(train_data_list_f, test_data_list_f)

    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
    test_accuracy_list.append(test_accuracy)

    # allpre = list(map(lambda x: int(x[0]) + x[1], zip(pre, allpre)))
    # train_feature_list, test_feature_list = TextFeatures(train_data_list, nnews_list, feature_words, flag)
    # TextClassifieracc(train_feature_list, test_feature_list, train_class_list, flag)

    print (test_accuracy_list)
    print ("finished")

    # 结果评价，如此处报错可暂时忽略
    # plt.figure()
    # plt.plot(deleteNs, test_accuracy_list)
    # plt.title('Relationship of deleteNs and test_accuracy')
    # plt.xlabel('deleteNs')
    # plt.ylabel('test_accuracy')
    # plt.savefig('result.png')