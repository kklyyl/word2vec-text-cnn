
import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
 
if __name__ == "__main__":
    corpus=["我 来到 北京 清华大学",#第一类文本切词后的结果，词之间以空格隔开
        "他 来到 了 网易 杭研 大厦"]#第二类文本的切词结果

    cv=CountVectorizer(token_pattern='[\u4e00-\u9fa5_a-zA-Z0-9]{1,}')  #匹配单个文字
    cv_fit=cv.fit_transform(corpus)
    voacb_list=cv.get_feature_names()
    print("构建好的词典",len(voacb_list))
    transformer = TfidfTransformer() #该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(cv_fit)#输入词频矩阵转化为对应的tf-idf值
    result_arr = tfidf.toarray()
    #print (len(result_arr),result_arr.shape)

    for i in range(len(result_arr)):
        print("-------这里输出第",i,"行文本的词语tf-idf权重------")
        for j in range(len(voacb_list)):
            print (voacb_list[j],result_arr[i][j])
