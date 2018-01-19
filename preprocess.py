#coding: utf-8

import jieba
import MySQLdb
import jieba.analyse


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

'''
以下为从数据库读取新闻，滤去空信息，并将写成文件放在/test/8/文件夹下作为新的测试文件，因为跑过一遍可以先不用跑
'''

news_list = []
nnews_list=[]
keywords = []
stoplist = {}.fromkeys([line.strip() for line in open("./stopwords_cn.txt")])
num=1
ReadFromSQL(news_list)
for i in news_list:
    txt_path='./data/test/8/'+'8_'+str(num)+'.txt'
    f=open(txt_path,'w')
    if i=="":
        continue
    print ("testdata"+str(num))
    print (i)
    nnews_list.append(i)
    f.write(i)
    f.close()
    num+=1
Eventex(nnews_list)

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
