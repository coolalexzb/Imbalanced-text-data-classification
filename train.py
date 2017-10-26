#coding: utf-8
import os
import jieba
import jieba.analyse


'''
    数据预处理，原函数因为样本数量有限，对样本做了训练集和测试集的划分，此处不需要划分，
    可以详细看数据处理流程，不懂的地方可用print输出看结果
'''
def TextProcessingsep(train_path,test_size):
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
            if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                    all_words_list[t]) < 5:
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

def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r') as fp:
        for line in fp.readlines():
            word = line.strip().decode("utf-8")
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set




train_path='./training/'
stopwords_file = './stopwords_cn.txt'
stopwords_set = MakeWordsSet(stopwords_file)
feature, train_data_list, train_class_list = TextProcessingsep(train_path,test_size=0.2)
file_traindata = open("train_data.txt",'wb')
file_trainclass = open("train_class.txt",'w')
feature_words = open("feature_words.txt",'w')

for i in train_data_list:
    print i
    break

for i in train_data_list:
    for j in i:
        j.encode("utf-8")
        print type(j)
        print "".join(j)

    break
print len(train_data_list)
# for i in feature:
#     print i
for item in train_data_list:
    #temp = str(item).strip('[').strip(']')
    #file_traindata.write(temp.encode('gbk')+"\n")
    for i in item:
        file_traindata.write("%s"%i.encode("utf-8"))
        file_traindata.write(" ")
    file_traindata.write('\n')
file_traindata.close()


for item in train_class_list:
    temp = str(item).strip('[').strip(']').replace(' ', '')
    file_trainclass.write(temp + '\n')
file_trainclass.close()

for item in feature:
    temp = str(item).strip('[').strip(']').replace(' ', '')
    feature_words.write(temp + '\n')
feature_words.close()

