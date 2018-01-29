#coding: utf-8
import os
import jieba
import jieba.analyse
from SFeature import *
import synonyms
import re
import math
'''
    数据预处理，原函数因为样本数量有限，对样本做了训练集和测试集的划分，此处不需要划分，
    可以详细看数据处理流程，不懂的地方可用print输出看结果
'''
def TextProcessingsep(train_path):
    folder_list = os.listdir(train_path)
    train_data_list = []
    train_class_list = []
    feature_words = []
    word_dic ={}
    doc ={}
    pattern = u'[\w|.]+'.encode('utf-8')
    p = re.compile(pattern)
    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(train_path, folder)
        files = os.listdir(new_folder_path)
        doc[int(folder)] = len(files)
        cata_data_list = []
        # 类内循环
        j = 1
        for file in files:
            if j > 5000: # 每类text样本数最多100
                break
            with open(os.path.join(new_folder_path, file).replace('\\','/'), 'r',encoding='utf-8') as fp:
               raw = fp.read()
            ## jieba分词
            # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
            word_cut = jieba.cut(raw, cut_all=False) # 精确模式，返回的结构是一个可迭代的genertor
            word_list = list(word_cut) # genertor转化为list，每个词unicode格式
            # jieba.disable_parallel() # 关闭并行分词模式
            train_data_list.append(word_list)
            train_class_list.append(folder)
            j += 1
            cata_data_list.append(word_list)

        all_words_dict = {}
        for word_list in cata_data_list:
            for word in word_list:
                if all_words_dict.__contains__(word):
                    all_words_dict[word] += 1
                else:
                    all_words_dict[word] = 1
        # key函数利用词频进行降序排序
        all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)  # 内建函数sorted参数需为list
        all_words_list = []
        for i in all_words_tuple_list:
            all_words_list.append(i[0])
        # 选取特征词
        n = 1
        for t in range(0, len(all_words_list), 1):
            if n > 200:  # feature_words的维度每类100个
                break
            # print all_words_list[t]
            if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                    all_words_list[t]) < 5 and len(p.findall(all_words_list[t].encode('utf-8'))) == 0:
                feature_words.append(all_words_list[t])
                n += 1

        kind_words_dict = {}
        for word_list in cata_data_list:
            for word in set(word_list):
                if kind_words_dict.__contains__(word):
                    kind_words_dict[word] += 1
                else:
                    kind_words_dict[word] = 1
        for k in kind_words_dict:
            kind_words_dict[k] =(int(folder) , kind_words_dict[k])
        word_dic[int(folder)] = kind_words_dict


    ## 划分训练集和测试集
    # train_data_list, test_data_list, train_class_list, test_class_list = sklearn.cross_validation.train_test_split(data_list, class_list, test_size=test_size)
    #data_class_list = zip(data_list, class_list)
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

    last_word_dic = {}
    for k in word_dic:
        for word in word_dic[k]:
            if word in last_word_dic:
                last_word_dic[word].append((word_dic[k][word]))
            else:
                last_word_dic[word] = [(word_dic[k][word])]
    fea_word = []
    for k in word_dic:
        cur_word_dic = {}
        for word in word_dic[k]:
            if word in last_word_dic.keys():
                cur_word_dic[word] = last_word_dic[word]
        #score_set = cal_score(cur_word_dic ,doc)
        score_set = chi_square(cur_word_dic , doc , k)
        score_words_tuple_list = sorted(score_set.items(), key=lambda f: f[1], reverse=True)  # 内建函数sorted参数需为list
        score_words_list = []
        for i in score_words_tuple_list:
            score_words_list.append(i[0])
        # 选取特征词
        n = 1
        for t in range(0, len(score_words_list), 1):
            if n > 200:  # feature_words的维度每类100个
                break
            # print all_words_list[t]
            if not score_words_list[t].isdigit() and score_words_list[t] not in stopwords_set and 1 < len(
                    score_words_list[t]) < 5 and len(p.findall(score_words_list[t].encode('utf-8'))) == 0:
                fea_word.append(score_words_list[t])
                n += 1


    return fea_word, train_data_list,train_class_list


def cal_score(word_dic ,doc):
    score_set = {}
    all_doc_num = 0
    for k in doc:
        all_doc_num += doc[k]
    for k in word_dic:
        score = 0
        for i in doc:
            t = 0
            rt = 0
            Ci = 0
            for item in word_dic[k]:
                t += item[1]
                if item[0] == i:
                    Ci += item[1]
                else :
                    rt += item[1]
            score += (Ci / t) / ((doc[i] - Ci) / doc[i] + rt / (all_doc_num - doc[i]) + 1)

        score_set[k] = score
    return score_set

def chi_square(word_dic , doc , label):
    score_set = {}
    for k in word_dic :
        A = 0
        B = 0
        for item in word_dic[k] :
            if item[0] == label :
                A += item[1]
            else :
                B += item[1]
        C = doc[label] - A
        tmp = 0
        for docnum in doc :
            if docnum != label :
                tmp += doc[docnum]
        D = tmp - B
        score =  pow((A * D - B * C) , 2) / (A + B) * (C + D)
        score_set[k] = score
    return score_set

def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r' , encoding='utf-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set

if __name__ == '__main__':
    flag = "svm"
    train_path='./newdata/training/'
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)
    feature, train_data_list, train_class_list = TextProcessingsep(train_path)
    file_traindata = open("train_data.txt",'w',encoding='utf-8')
    file_trainclass = open("train_class.txt",'w',encoding='utf-8')
    feature_words = open("feature_words.txt",'w',encoding='utf-8')
    matrix_feature = open("matrix_fea.txt",'w',encoding='utf-8')

    feature_nearby = []
    for word in feature:
        tmp_nearby = synonyms.nearby(word)
        nearby = list((words, rank) for words, rank in zip(tmp_nearby[0], tmp_nearby[1]))
        real_nearby = word
        for item in nearby:
            if item[1] >= 0.61:
                real_nearby += " " + item[0]
            else:
                break
        feature_nearby.append(real_nearby)

    fea_tmp = [wordset.split(" ") for wordset in feature_nearby]                #str to list
    train_feature_list = TextFeatures(train_data_list, fea_tmp, flag)

    for item in train_data_list:
        for i in item:
            file_traindata.write(i)
            file_traindata.write(" ")
        file_traindata.write('\n')
    file_traindata.close()


    for item in train_class_list:
        temp = str(item).strip('[').strip(']').replace(' ', '')
        file_trainclass.write(temp + '\n')
    file_trainclass.close()

    for item in feature_nearby:
        temp = str(item).strip('[').strip(']')
        feature_words.write(temp + '\n')
    feature_words.close()

    for item in train_feature_list:
        temp = str(item).strip('[').strip(']').replace(' ', '')
        matrix_feature.write(temp + '\n')
    matrix_feature.close()



