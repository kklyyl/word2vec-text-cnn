#encoding:utf-8
import logging
import time
import codecs
import sys
import re
import jieba
from gensim.models import word2vec

class TextConfig():

    embedding_size=100     #dimension of word embedding
    vocab_size=8000        #number of vocabulary
    pre_trianing = None   #use vector_char trained by word2vec

    seq_length=600         #max length of sentence
    num_classes=34         #number of labels

    num_filters=128        #number of convolution kernel
    filter_sizes=[2,3,4]   #size of convolution kernel


    keep_prob=0.5          #droppout
    lr= 1e-3               #learning rate
    lr_decay= 0.9          #learning rate decay
    clip= 6.0              #gradient clipping threshold
    l2_reg_lambda=0.01     #l2 regularization lambda

    num_epochs=10          #epochs
    batch_size=64         #batch_size
    print_per_batch =100   #print result

    train_filename='./testdata/train.txt' 
    test_filename='./testdata/test.txt' 
    val_filename='./testdata/val.txt' 
    vocab_filename='./testdata/vocab.txt'        #vocabulary
    vector_word_filename='./testdata/vector_word.txt'  #vector_word trained by word2vec
    vector_word_npz='./testdata/vector_word.npz'   # save vector_word to numpy file

re_han= re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)") # the method of cutting text by punctuation

class Get_Sentences(object):
    '''

    Args:
         filenames: a list of train_filename,test_filename,val_filename
    Yield:
        word:a list of word cut by jieba

    '''

    def __init__(self,filenames):
        self.filenames= filenames

    def __iter__(self):
        for filename in self.filenames:
            with codecs.open(filename, 'r', encoding='utf-8') as f:
                for _,line in enumerate(f):
                    try:
                        line=line.strip()
                        line=line.split('\t')
                        assert len(line)==2
                        blocks=re_han.split(line[1])
                        word=[]
                        for blk in blocks:
                            if re_han.match(blk):
                                word.extend(jieba.lcut(blk))
                        yield word
                    except:
                        pass

def train_word2vec(filenames):
    
    '''
    use word2vec train word vector
    argv:
        filenames: a list of train_filename,test_filename,val_filename
    return: 
        save word vector to config.vector_word_filename

    '''
    t1 = time.time()
    sentences = Get_Sentences(filenames)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=1, workers=6)
    model.wv.save_word2vec_format(config.vector_word_filename, binary=False)
    print('-------------------------------------------')
    print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))

if __name__ == '__main__':
    print("================test: word2vec ==============================")
    config=TextConfig()
    
    filenames=[config.train_filename,config.test_filename,config.val_filename]
    train_word2vec(filenames)

