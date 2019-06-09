# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:30:45 2019

@author: liweimin
"""
#选择keras的原因
#允许简单而快速的原型设计（由于用户友好，高度模块化，可扩展性）。
#同时支持卷积神经网络和循环神经网络，以及两者的组合。
#在 CPU 和 GPU 上无缝运行。

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from keras_contrib.layers.crf import CRF
import matplotlib.pyplot as plt
import os
import sys

# 只显示 warning 和 Error 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LSTMNER:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_path = os.path.join(cur, 'data/train.txt')
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        self.embedding_file = os.path.join(cur, 'model/token_vec_300.bin')
        self.model_path = os.path.join(cur, 'model/tokenvec_bilstm2_crf_model_20.h5')
        self.datas, self.word_dict = self.build_data()
        self.class_dict ={
#ner序列
#                         'O':0,
#                         'tre-I': 1,
#                         'tre-B': 2,
#                         'bod-B': 3,
#                         'bod-I': 4,
#                         'sym-I': 5,
#                         'sym-B': 6,
#                         'tes-B': 7,
#                         'tes-I': 8,
#                         'dis-I': 9,
#                         'dis-B': 10,
#                         'nt-B' : 11,
#                         'nt-I' : 12,
#                         'nr-B' : 13,
#                         'nr-I' : 14
#分词序列
#                         'B': 0,
#                         'M': 1,
#                         'E': 2,
#                         'S': 3
#词性标注序列        
                         'a': 0,
                         'Ag': 1,
                         'ad': 2,
                         'an': 3,
                         'b': 4,
                         'c': 5,
                         'd': 6,
                         'Dg':7,
                         'e': 8,
                         'f': 9,
                         'g':10,
                         'h':11,
                         'i': 12,
                         'j': 13,
                         'k':14,
                         'l': 15,
                         'm': 16,
                         'n': 17,
                         'Ng':18,
                         'ng':19,
                         'nr' : 20,
                         'ns' : 21,
                         'nt' : 22,
                         'nx' : 23,
                         'nz': 24,
                         'o':25,
                         'p': 26,
                         'q': 27,
                         'r': 28,
                         's': 29,
                         'Tg':30,
                         't': 31,
                         'u': 32,
                         'v': 33,
                         'vd': 34,
                         'vn': 35,
                         'Vg':36,
                         'w' : 37,
                         'x': 38,
                         'y': 39,
                         'z': 40,
                         '$$_': 41,
                         'nrx': 42,
                         'sub>': 43,
                         'm<': 44,
                         'sup>': 45,
                         'Bg': 46

                        }
#初始使用论文“Neural Architectures for Named Entity Recognition”提供的参数，然后根据模型效果进行超参数调参        
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 10 #10次epoch后准确率基本上就不在变化了
        self.BATCH_SIZE = 128
        self.NUM_CLASSES = len(self.class_dict)
        self.VOCAB_SIZE = len(self.word_dict)
        self.TIME_STAMPS = 150   #时间步
        self.embedding_matrix = self.build_embedding_matrix()

    #构造数据集
    def build_data(self):
        datas = []
        sample_x = []
        sample_y = []
        vocabs = {'UNK'}
        for line in open(self.train_path,'r', encoding='UTF-8'): #对竖着的文本数据按 。进行切分，循环采集每一句话
            line = line.rstrip().split('\t')  #Python rstrip() 删除 string 字符串末尾的指定字符（默认为空格）
            if not line:
                continue
            char = line[0]
            if not char:
                continue
            cate = line[-1] #category为line的最后一个
            sample_x.append(char)
            sample_y.append(cate)
            vocabs.add(char)
            if char in ['。','？','！']:   #如果char里有表示句子终止的符号，则认为一句话结束，加入datas中
                datas.append([sample_x, sample_y])
                sample_x = []
                sample_y = []
        word_dict = {wd:index for index, wd in enumerate(list(vocabs))}
        self.write_file(list(vocabs), self.vocab_path)
        return datas, word_dict

    #将数据转换成keras所需的格式
    def modify_data(self):
        x_train = [[self.word_dict[char] for char in data[0]] for data in self.datas]
        y_train = [[self.class_dict[label] for label in data[1]] for data in self.datas]
        x_train = pad_sequences(x_train, self.TIME_STAMPS)
        y = pad_sequences(y_train, self.TIME_STAMPS)
        y_train = np.expand_dims(y, 2)
        return x_train, y_train

    #保存字典文件
    def write_file(self, wordlist, filepath):
        with open(filepath, 'w+', encoding='UTF-8') as f:
            f.write('\n'.join(wordlist))

    #加载预训练词向量
    def load_pretrained_embedding(self):
        embeddings_dict = {}
        with open(self.embedding_file, 'r', encoding='UTF-8') as f:
            for line in f:
                values = line.strip().split(' ')
                if len(values) < 300:  #20028维 300列
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
        print('Found %s word vectors.' % len(embeddings_dict))
        return embeddings_dict

    #加载词向量矩阵
    def build_embedding_matrix(self):
        embedding_dict = self.load_pretrained_embedding()
        embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
        for word, i in self.word_dict.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    #使用预训练向量进行模型训练
    def tokenvec_bilstm2_crf_model(self):
        model = Sequential()
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=True)
        model.add(embedding_layer)  # 150，300
        model.add(Bidirectional(LSTM(128, return_sequences=True)))    # 150，256    2*128=256
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))  #150，128        2*64=128
        model.add(Dropout(0.5))
#        model.add(Bidirectional(LSTM(32, return_sequences=True)))  #150，64        2*32=64
#        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(self.NUM_CLASSES)))     # 150，15   15个class_dict
        crf_layer = CRF(self.NUM_CLASSES, sparse_target=True)     # 150，15   15个class_dict  crf计算分数最高的class
        model.add(crf_layer)
        model.compile('adam', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
        model.summary()
        return model

    #训练模型
    def train_model(self):
        x_train, y_train = self.modify_data()
        model = self.tokenvec_bilstm2_crf_model()
        history = model.fit(x_train[:], y_train[:], validation_split=0.2, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS)
        self.draw_train(history)
        model.save(self.model_path)
        return model

    #绘制训练曲线
    def draw_train(self, history):
        # Plot training & validation accuracy values
        plt.plot(history.history['crf_viterbi_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()
       

if __name__ == '__main__':
    ner = LSTMNER()
    ner.train_model()