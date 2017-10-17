
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
#import MySQLdb
import jieba.analyse
from sklearn.svm import SVC
import MySQLdb

#生成停用词字典
def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r') as fp:
        for line in fp.readlines():
            word = line.strip().decode("utf-8")
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set

#数据预处理，原始函数，可先忽略
def TextProcessing(folder_path, test_size=0.2):
    folder_list = os.listdir(folder_path.decode('utf-8'))
    data_list = []
    class_list = []

    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path.decode('utf-8'))
        # 类内循环
        j = 1
        for file in files:
            if j > 100: # 每类text样本数最多100
                break
            with open(os.path.join(new_folder_path, file).replace('\\','/'), 'r') as fp:
               raw = fp.read()
            #print raw
            ## --------------------------------------------------------------------------------
            ## jieba分词
            # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
            word_cut = jieba.cut(raw, cut_all=False) # 精确模式，返回的结构是一个可迭代的genertor
            word_list = list(word_cut) # genertor转化为list，每个词unicode格式
            # jieba.disable_parallel() # 关闭并行分词模式
            # print word_list
            ## --------------------------------------------------------------------------------
            data_list.append(word_list)
            class_list.append(folder.decode('utf-8'))
            j += 1



    ## 划分训练集和测试集
    # train_data_list, test_data_list, train_class_list, test_class_list = sklearn.cross_validation.train_test_split(data_list, class_list, test_size=test_size)
    data_class_list = zip(data_list, class_list)

    '''
    for i in data_class_list:
        print "data_class_list: "
        print i
    '''
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*test_size)+1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if all_words_dict.has_key(word):
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True) # 内建函数sorted参数需为list
    all_words_list = list(zip(*all_words_tuple_list)[0])

    '''
    print "all: "
    for i in all_words_list:
        print i
    '''

    j=1;
    # for i in test_data_list :
    #     print "testdata"+str(j)
    #     print"".join(i)
    #     j+=1
    '''
    for i in train_data_list:
        print "traindata"
        print "".join(i)
    '''
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


'''
    数据预处理，原函数因为样本数量有限，对样本做了训练集和测试集的划分，此处不需要划分，
    可以详细看数据处理流程，不懂的地方可用print输出看结果
'''
def TextProcessingsep(train_path,test_path, test_size):
    folder_list = os.listdir(train_path.decode('utf-8'))
    train_data_list = []
    train_class_list = []
    test_data_list=[]
    test_class_list=[]
    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(train_path, folder)
        files = os.listdir(new_folder_path.decode('utf-8'))
        # 类内循环
        j = 1
        for file in files:
            if j > 100: # 每类text样本数最多100
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


    folder_list = os.listdir(test_path.decode('utf-8'))
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


    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if all_words_dict.has_key(word):
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True) # 内建函数sorted参数需为list
    all_words_list = list(zip(*all_words_tuple_list)[0])

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
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

#提取特征词
def words_dict(all_words_list, deleteN, stopwords_set=set()):
    # 选取特征词
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000: # feature_words的维度1000
            break
        # print all_words_list[t]
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1<len(all_words_list[t])<5:
            feature_words.append(all_words_list[t])
            n += 1


    for i in feature_words:
       print "fea:     "+i
    print "--------------finish--------------"
    return feature_words







#916从每类新闻训练集中抽取指定数量的feature words
def words_dict_new(train_path, stopwords_set=set()):
    # 统计词频放入all_words_dict
    feature_words = []
    for i in range(1,8,1):
        # folder_list = os.listdir(train_path.decode('utf-8'))
        train_data_list = []
        train_class_list = []
        # 类间循环
        # for folder in folder_list:
        new_folder_path = os.path.join(train_path, str(i))
        files = os.listdir(new_folder_path.decode('utf-8'))
            # 类内循环
        j = 1
        for file in files:
            if j > 2000:  # 每类text样本数最多100
                break
            with open(os.path.join(new_folder_path, file).replace('\\', '/'), 'r') as fp:
                raw = fp.read()
            word_cut = jieba.cut(raw, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
            word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
            train_data_list.append(word_list)
            train_class_list.append(file.decode('utf-8'))
            j += 1
        all_words_dict = {}
        for word_list in train_data_list:
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
            if n > 100: # feature_words的维度1000
                break
            # print all_words_list[t]
            if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1<len(all_words_list[t])<5:
                feature_words.append(all_words_list[t])
                n += 1

    #
    # for i in feature_words:
    #    print "fea:     "+i
    # print "--------------finish--------------"
    return feature_words

#提取新闻特征，即事件抽取
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
        elif flag == 'svm':
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []
        ## -----------------------------------------------------------------------------------
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]

    return train_feature_list, test_feature_list


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag='nltk'):
    ## -----------------------------------------------------------------------------------
    '''
    print   (type(test_class_list))
    for word in train_class_list:
        print "train:   " + word
    for word in test_class_list:
        print "test:    " + word
    '''
    y=[]
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
        y=x.tolist()
        # print y[0]
        # for test_feature in test_feature_list:
        #     print classifier.predict(test_feature)[0]

        test_accuracy = classifier.score(test_feature_list, test_class_list)
    elif flag=='svm':
        classifier=SVC()
        classifier.fit(train_feature_list,train_class_list)
        print "presvm:"
        print classifier.predict(test_feature_list)

        test_accuracy=classifier.score(test_feature_list, test_class_list)
    else:
        test_accuracy = []

    # print test_accuracy

    return test_accuracy
    # return y

def TextClassifieracc(train_feature_list, test_feature_list, train_class_list, flag='nltk'):

    if flag == 'nltk':
        ## nltk分类器
        train_flist = zip(train_feature_list, train_class_list)
        #test_flist = zip(test_feature_list, test_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        # print classifier.classify_many(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.classify(test_feature),
        # print ''
        #test_accuracy = nltk.classify.accuracy(classifier, test_flist)
    if flag == 'sklearn':
        ## sklearn分类器
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        # print "pre: "
        # print classifier.predict(test_feature_list)
        #for test_feature in test_feature_list:
        #     print classifier.predict(test_feature)[0]

        test_accuracy = classifier.score(test_feature_list, test_class_list)
    else:
       test_accuracy = []

    print test_accuracy

    return test_accuracy


# 从数据库读新闻
def ReadFromSQL(news_list):
    conn = MySQLdb.connect(
        host='localhost',
        port=3306,
        user='root',
        passwd='root',
        db='crawler',
    )
    cur = conn.cursor()
    conn.set_character_set('utf8')
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')
    count=cur.execute("select passage from crawler_sina")

    results=cur.fetchall()
    result=list(results)
    for i in result:
        news_list.append("".join(i))

    cur.close()
    conn.commit()
    conn.close()

def Eventex(news_list):
    for i in news_list:
        segs = jieba.cut(i, cut_all=False)
        segs = [word.encode('utf-8') for word in list(segs)]
        segs = [word for word in list(segs) if word not in stoplist]
        tags=jieba.analyse.extract_tags("".join(segs), 10)
        #print " ".join(tags)
        keywords.append(" ".join(tags).encode('utf-8'))

def NewsfromDatabase(news_list):

    pass



if __name__ == '__main__':

    print "start"
    '''
    以下为从数据库读取新闻，滤去空信息，并将写成文件放在/test/8/文件夹下作为新的测试文件，因为跑过一遍可以先不用跑
    '''

    '''
    news_list = []
    nnews_list=[]
    keywords = []
    stoplist = {}.fromkeys([line.strip() for line in open("/usr/local/lib/python2.7/dist-packages/adascrawler/newscrawler/Naive-Bayes-Classifier-master/Naive-Bayes-Classifier-master/stopwords_cn.txt")])
    num=1
    ReadFromSQL(news_list)
    for i in news_list:
        txt_path='/usr/local/lib/python2.7/dist-packages/adascrawler/newscrawler/Naive-Bayes-Classifier-master/Naive-Bayes-Classifier-master/data/test/8/'+'8_'+str(num)+'.txt'
        f=open(txt_path,'w')
        if i=="":
            continue
        print "testdata"+str(num)
        print i
        nnews_list.append(i)
        f.write(i)
        f.close()
        num+=1
    Eventex(nnews_list)
    '''
    ''' 文本预处理,注释部分之前是用来测试搜狗语料库的，效果还算满意
        非注释部分是新的数据集，新的数据集中将数据集分成训练集和测试集合
        新数据集数量是原数据集10倍
        原始搜集的数据里面各个数据集数量有较大偏差，为确保结果，将多余数据存入datax文件夹中
         '''
    #folder_path = 'F:/STUDY_DATA/Naive-Bayes-Classifier-master/Naive-Bayes-Classifier-master/Database/SogouC/Sample/'
    #folder_path = 'F:/STUDY_DATA/Naive-Bayes-Classifier-master/Naive-Bayes-Classifier-master/data/training/'
    #all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)

    train_path='/usr/local/lib/python2.7/dist-packages/adascrawler/newscrawler/Naive-Bayes-Classifier-master/Naive-Bayes-Classifier-master/data/training'
    test_path='/usr/local/lib/python2.7/dist-packages/adascrawler/newscrawler/Naive-Bayes-Classifier-master/Naive-Bayes-Classifier-master/data/test'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessingsep(train_path,test_path,test_size=0.2)


    # 生成stopwords_set
    stopwords_file = '/usr/local/lib/python2.7/dist-packages/adascrawler/newscrawler/Naive-Bayes-Classifier-master/Naive-Bayes-Classifier-master/stopwords_cn.txt'
    # stopwords_file = 'F:/STUDY_DATA/Naive-Bayes-Classifier-master/Naive-Bayes-Classifier-master/stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    ## 文本特征提取和分类
    # flag = 'nltk'
    # flag = 'sklearn'    # 可修改此处更换分类器
    num=100
    flag='svm'
    deleteNs = range(0, 1000, 20)
    test_accuracy_list = []
    allpre=[0]*num
    count=0
    feature_words = words_dict_new(train_path, stopwords_set)
    #多组实验
    for deleteN in deleteNs:
        # feature_words = words_dict(all_words_list, deleteN)
        # feature_words = words_dict_new(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words, 'svm')

        '''
        for x in train_feature_list :
            print "train feature list:    "
            print x
        for x in test_feature_list :
            print "test_feature_list"
            print x
        '''
        count+=1
        # pre=TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, 'svm')
        # pre=list(pre)
        # allpre = list(map(lambda x: int(x[0]) + int(x[1]), zip(pre, allpre)))
        test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
        test_accuracy_list.append(test_accuracy)
        # allpre = list(map(lambda x: int(x[0]) + x[1], zip(pre, allpre)))
        #train_feature_list, test_feature_list = TextFeatures(train_data_list, nnews_list, feature_words, flag)
        #TextClassifieracc(train_feature_list, test_feature_list, train_class_list, flag)

    #除数多组实验的准确率
    print test_accuracy_list


    '''c=[count]*num
    print count,allpre
    allpre=list(map(lambda x: int(x[0]) / int(x[1]), zip(allpre, c)))
    all=[]
    print allpre
    d={'1':"财经",'2':"科技",'3':"汽车",'4':"房产",'5':"体育",'6':"娱乐",'7':"其他"}
    for i in allpre:
        i=str(i)
        all.append(d[i])
    # print all
    conn = MySQLdb.connect(
        host='localhost',
        port=3306,
        user='root',
        passwd='root',
        db='crawler',
    )
    cur = conn.cursor()
    conn.set_character_set('utf8')
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')
    tmp=zip(news_id,all,news_id)
    for i in tmp:
            #print i
        try:
            sqli = "insert into " + "crawler_sina_classifier " + "values(%s,%s,%s)"
            # print "_________________________"
            cur.execute(sqli,i)
            print "================================"
        except Exception,e:
            print e
            c=0
                
    cur.close
    conn.commit()
    conn.close()
'''

    # 结果评价，如此处报错可暂时忽略
    # plt.figure()
    # plt.plot(deleteNs, test_accuracy_list)
    # plt.title('Relationship of deleteNs and test_accuracy')
    # plt.xlabel('deleteNs')
    # plt.ylabel('test_accuracy')
    # plt.savefig('result.png')

    print "finished"