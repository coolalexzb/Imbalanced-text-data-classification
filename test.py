#coding: utf-8
import os
import time
import random
import jieba
import nltk
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import jieba.analyse
from sklearn.svm import SVC
from sklearn.decomposition import PCA,NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer ,HashingVectorizer,CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import GaussianNB



'''
    数据预处理，原函数因为样本数量有限，对样本做了训练集和测试集的划分，此处不需要划分，
    可以详细看数据处理流程，不懂的地方可用print输出看结果
'''
def TextProcessingsep(test_path, test_size):
    folder_list = os.listdir(test_path.decode('utf-8'))
    test_data_list= []
    test_class_list= []
    for folder in folder_list:
        new_folder_path = os.path.join(test_path, folder)
        #print new_folder_path
        files = os.listdir(new_folder_path.decode('utf-8'))
        # 类内循环
        j = 1
        for file in files:
            if j > 100:  # 每类text样本数最多100
                break
            with open(os.path.join(new_folder_path, file).replace('\\', '/'), 'r') as fp:
                raw = fp.read()
            # print raw
            ## --------------------------------------------------------------------------------
            ## jieba分词
            # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
            word_cut = jieba.cut(raw, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
            word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
            # jieba.disable_parallel() # 关闭并行分词模式
            # print word_list
            ## --------------------------------------------------------------------------------
            test_data_list.append(word_list)
            test_class_list.append(folder.decode('utf-8'))
            j += 1
        '''
        for i in news_list:
            segs = jieba.cut(i, cut_all=False)
            segs = [word.encode('utf-8') for word in list(segs)]
            segs = [word for word in list(segs) if word not in stoplist]
            tags = jieba.analyse.extract_tags("".join(segs), 10)
            # print " ".join(tags)
            keywords.append(" ".join(tags).encode('utf-8'))
        '''

    # 统计词频放入all_words_dict
    return test_data_list, test_class_list

#生成停用词字典
def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r') as fp:
        for line in fp.readlines():
            word = line.strip().decode("utf-8")
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set




def remove_stopwords(train_data_list, test_data_list,stopwords_set):
    train_data_filter=[]
    test_data_filter=[]
    for i in train_data_list:
        filter = []
        for j in i:
            if j not in stopwords_file:
                filter.append(j)
        train_data_filter.append(filter)
    for i in test_data_list:
        filter = []
        for j in i:
            if j not in stopwords_file:
                filter.append(j)
        test_data_filter.append(filter)
    return train_data_filter , test_data_filter


def TextFeatures(train_data_list, test_data_list, feature_words, flag='sklearn'):#修改一个
    def text_features(text, feature_words):
        text_words = set(text)
        ## -----------------------------------------------------------------------------------
        if flag == 'nltk':
            ## nltk特征 dict
            features = {word:1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            ## sklearn特征 list
            features = [1 if word in text_words else 0 for word in feature_words]
            '''
                       print "content: "
                       for i in text:
                           print "".join(i),
                       print '---------------------------------------------'
                       print "res"
                       for i in feature_words:
                           if i in text_words:
                               print i+"   ",
                       print '----------------------------------------------'
                       '''
        elif flag == 'svm':
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []
        ## -----------------------------------------------------------------------------------
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]

    return train_feature_list, test_feature_list



def dimension_reduce(train_data_list_f, test_data_list_f):

    corpus_train=[]
    corpus_test=[]
    train_feature_list=[]
    test_feature_list=[]

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

    tv = TfidfVectorizer(sublinear_tf=True,max_df=0.5,)
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
    #pca_test = PCA(n_components=50)
    #test_feature_list = pca_test.fit_transform(tfidf_test_2)

    '''
    train_nmf = NMF(n_components=10, random_state=1,
              beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
              l1_ratio=.5).fit(train_cop_tfidf)
    test_nmf = NMF(n_components=10, random_state=1,
                    beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
                    l1_ratio=.5).fit(test_cop_tfidf)
    '''
    #print(type(train_cop_tfidf))
    #print(type(train_feature_list))
    #print(type(test_feature_list))

    return tfidf_train_2 , tfidf_test_2
    #return train_nmf, test_nmf
    #return train_feature_list, test_feature_list


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag='nltk'):
    ## -----------------------------------------------------------------------------------
    '''
    print   (type(test_class_list))
    for word in train_class_list:
        print "train:   " + word
    for word in test_class_list:
        print "test:    " + word
    '''
    #y=[]
    if flag == 'nltk':
        ## nltk分类器
        train_flist = zip(train_feature_list, train_class_list)
        test_flist = zip(test_feature_list, test_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        #print classifier.classify_many(test_feature_list)
        #for test_feature in test_feature_list:
        #     print classifier.classify(test_feature),

        test_accuracy = nltk.classify.accuracy(classifier, test_flist)
    elif flag == 'sklearn':
        ## sklearn分类器
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        print "pre: "
        x=classifier.predict(test_feature_list)
        # print test_feature_list
        print classifier.predict(test_feature_list)
        #y=x.tolist()
        # print y[0]
        # for test_feature in test_feature_list:
        #     print classifier.predict(test_feature)[0]

        test_accuracy = classifier.score(test_feature_list, test_class_list)
    elif flag=='svm':
        classifier=SVC(kernel = 'linear')
        classifier.fit(train_feature_list,train_class_list)
        print "presvm:"
        print classifier.predict(test_feature_list)

        test_accuracy=classifier.score(test_feature_list, test_class_list)
    elif flag=='GussNB':
        classifier=GaussianNB().fit(train_feature_list, train_class_list)
        print "pre: "
        print classifier.predict(test_feature_list)
    else:
        test_accuracy = []

    # print test_accuracy

    return test_accuracy

def TextProcessingsepx(train_path,test_size):
    folder_list = os.listdir(train_path.decode('utf-8'))
    train_data_list = []
    train_class_list = []
    feature_words = []
    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(train_path, folder)
        files = os.listdir(new_folder_path.decode('utf-8'))
        cata_data_list = []
        # 类内循环
        j = 1
        for file in files:
            if j > 1000: # 每类text样本数最多100
                break
            with open(os.path.join(new_folder_path, file).replace('\\','/'), 'r') as fp:
               raw = fp.read()
            # print raw
            ## --------------------------------------------------------------------------------
            ## jieba分词
            # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
            word_cut = jieba.cut(raw, cut_all=False) # 精确模式，返回的结构是一个可迭代的genertor
            word_list = list(word_cut) # genertor转化为list，每个词unicode格式
            # jieba.disable_parallel() # 关闭并行分词模式
            # print word_list
            ## --------------------------------------------------------------------------------
            train_data_list.append(word_list)
            train_class_list.append(folder.decode('utf-8'))
            j += 1
            cata_data_list.append(word_list)

        all_words_dict = {}
        for word_list in cata_data_list:
            for word in word_list:
                if all_words_dict.has_key(word):
                    all_words_dict[word] += 1
                else:
                    all_words_dict[word] = 1
        # key函数利用词频进行降序排序
        all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)  # 内建函数sorted参数需为list
        all_words_list = list(zip(*all_words_tuple_list)[0])
        # 选取特征词

        n = 1
        for t in range(0, len(all_words_list), 1):
            if n > 100:  # feature_words的维度1000
                break
            # print all_words_list[t]
            if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t])< 5:
                feature_words.append(all_words_list[t])
                n += 1

    ## 划分训练集和测试集
    # train_data_list, test_data_list, train_class_list, test_class_list = sklearn.cross_validation.train_test_split(data_list, class_list, test_size=test_size)
    #data_class_list = zip(data_list, class_list)


    #for i in data_class_list:
    #    print "data_class_list: "
    #    print i

    #random.shuffle(data_class_list)
    #index = int(len(data_class_list)*test_size)+1
    #train_list = data_class_list[index:]
    #test_list = data_class_list[:index]
    #train_data_list, train_class_list = zip(*train_list)
    #test_data_list, test_class_list = zip(*test_list)

    '''
        for i in news_list:
            segs = jieba.cut(i, cut_all=False)
            segs = [word.encode('utf-8') for word in list(segs)]
            segs = [word for word in list(segs) if word not in stoplist]
            tags = jieba.analyse.extract_tags("".join(segs), 10)
            # print " ".join(tags)
            keywords.append(" ".join(tags).encode('utf-8'))
        '''


    #print "all: "
    #for i in all_words_list:
    #    print i

    #输出测试新闻的内容，可注释
    '''
    j=1;
    for i in test_data_list :
        print "testdata"+str(j)
        print"".join(i)
        j+=1
    '''
    '''
    for i in train_data_list:
        print "traindata"
        print "".join(i)
    '''
    return feature_words, train_data_list,train_class_list

if __name__ == '__main__':

    print "start"
    # train_data_list = []
    # train_class_list = []
    # #all_words_list = []
    # feature_words = []
    # file_traindata = open("train_data.txt",'r')
    # file_trainclass = open("train_class.txt",'r')
    # file_features = open("feature_words.txt",'r')
    #
    #
    # a =1
    # for lines in file_traindata:
    #     a=a+1
    #     temp =lines.strip().split(' ')
    #     # line = []
    #     # for i in temp:
    #     #     line.append(i)
    #     train_data_list.append(temp)
    # print a
    # for lines in file_trainclass:
    #     train_class_list.append(int(lines))
    # for lines in file_features:
    #     feature_words.append(lines.strip().decode())
    #
    # t = []
    # print ("assd")
    # for i in train_data_list:
    #     temp =[]
    #     for j in i:
    #         temp.append(j.decode('utf-8'))
    #     t.append(temp)
    # for i in t :
    #     for j in i :
    #         print type(j)
    #         print "".join(j)
    #     break
    #


    '''
       文本预处理,注释部分之前是用来测试搜狗语料库的，效果还算满意
       非注释部分是新的数据集，新的数据集中将数据集分成训练集和测试集合
       新数据集数量是原数据集10倍
       原始搜集的数据里面各个数据集数量有较大偏差，为确保结果，将多余数据存入datax文件夹中
    '''
    test_path='./test/'
    train_path = './training/'
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    feature_words, train_data_list, train_class_list = TextProcessingsepx(train_path,test_size=0.2)
    test_data_list, test_class_list = TextProcessingsep(test_path,test_size=0.2)

    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    ## 文本特征提取和分类
    # flag = 'sklearn'    # 可修改flag更换分类器
    num = 100
    flag = 'svm'
    deleteNs = range(0, 1000, 20)
    test_accuracy_list = []
    allpre = [0] * num
    count = 0
    train_data_list_f, test_data_list_f = remove_stopwords(train_data_list, test_data_list, stopwords_set)
    # print "=============="
    # for i in train_data_list_f:
    #     print i
    # print "==================="
    # for i in test_data_list_f:
    #     print i

    # 多组实验
    for deleteN in deleteNs:
        # feature_words = words_dict(all_words_list, deleteN)
        # feature_words = words_dict_new(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list_f, test_data_list_f, feature_words, 'svm')

        '''
        调用降维算法
        '''
        # train_feature_list, test_feature_list = dimension_reduce(train_data_list_f, test_data_list_f)


        '''
        for x in train_feature_list :
            print "train feature list:    "
            print x
        for x in test_feature_list :
            print "test_feature_list"
            print x
        '''
        count += 1
        # pre=TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, 'svm')
        # pre=list(pre)
        # allpre = list(map(lambda x: int(x[0]) + int(x[1]), zip(pre, allpre)))
        test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
        test_accuracy_list.append(test_accuracy)
        # allpre = list(map(lambda x: int(x[0]) + x[1], zip(pre, allpre)))
        # train_feature_list, test_feature_list = TextFeatures(train_data_list, nnews_list, feature_words, flag)
        # TextClassifieracc(train_feature_list, test_feature_list, train_class_list, flag)

    # 除数多组实验的准确率
    print test_accuracy_list

    print "finished"