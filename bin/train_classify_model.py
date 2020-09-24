#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
功能：利用预训练word2vec词向量模型+LSTM进行文本分类，本项目按需求分为三类，参数
"""

import numpy as np
import codecs
np.random.seed(1337)  
import pickle
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Input,InputLayer,Embedding, LSTM, Dense, Activation, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Dropout, Activation  
from keras.optimizers import SGD 
from keras.models import load_model
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import importlib
import sys
importlib.reload(sys)
from tensorflow.python.util import compat

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

corpus_word2vec_pkl_filename = '/data/model/query.all.split.pkl'   ###存放word2vec预训练模型
corpus_train_filename ='/data/dict/train.txt.all.labe'  ##放置的模型的训练集数据

# 参数设置
vocab_dim = 128  # 向量维度
maxlen = 20  #文本保留的最大长度
batch_size = 60
n_epoch = 220
input_length = maxlen
f = open(corpus_word2vec_pkl_filename, 'rb')  # 预先训练好的
index_dict = pickle.load(f) 
word_vectors = pickle.load(f)

classify_numb = 3  ##分类个数

##构造基本属性，从训练文件中获取训练数据和label
class TextSta:
    filename = ""
    def __init__(self,path):
        self.filename = path
    def sen(self):
        f1 = codecs.open(self.filename, "r", encoding="utf-8")
        sentences_list = []
        labels_list = []
        for line in f1:
            temp_list=line.strip().split("\t")
            if len(temp_list)==2:
               single_sen_list = temp_list[0].strip().split(" ")
               label = int(temp_list[1].split("__label__")[1])
            while "" in single_sen_list:
                single_sen_list.remove("")
            sentences_list.append(single_sen_list)
            labels_list.append(label)
        f1.close()
        return sentences_list,labels_list

def process_train_data(sentences_list,new_dic):
    new_sentences = []
    for line in sentences_list:
        new_sen = []
        for word in line:
            try:
                new_sen.append(new_dic[word])
            except:
                new_sen.append(0)
        new_sentences.append(new_sen)
    new_sentences = np.array(new_sentences)
    new_sentences_pad = sequence.pad_sequences(new_sentences,maxlen)
    new_sentences_pad = np.array(new_sentences_pad)
    return new_sentences_pad

def process_train_label(label_list):
    label_list = np_utils.to_categorical(label_list)
    label_list = np.array(label_list)
    return label_list

def get_train_data(sentences_list,labels_list):
    train_data,test_data,train_label,test_label = train_test_split(sentences_list,labels_list,test_size=0.15)
    train_data_emb = process_train_data(train_data,index_dict)
    test_data_emb = process_train_data(test_data,index_dict)
    train_label_hot = process_train_label(train_label)
    test_label_hot = process_train_label(test_label)
    return train_data_emb,test_data_emb,train_label_hot,test_label_hot


def export_savedmodel(model):
    model_path = "../class_model/" #自定义的生成模型文件夹的名字
    model_version = 1
    model_signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={'input': model.input}, outputs={'output': model.output})
    export_path = os.path.join(compat.as_bytes(model_path), compat.as_bytes(str(model_version)))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess=KTF.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        clear_devices=True,
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            model_signature
      })
    builder.save()
    print('模型已保存')


class LSTM_model:
    def __init__(self,p_n_symbols,p_embedding_weights):
        self.drop = Dropout(0.4)
        self.main_input = Input(shape=(input_length,), dtype='int32', name='main_input')
        self.embedding_x = Embedding(output_dim=vocab_dim,input_dim=p_n_symbols, mask_zero=True,weights=[p_embedding_weights],input_length=input_length)(self.main_input)
        self.lstm_out = LSTM(output_dim=50,activation='sigmoid',inner_activation='hard_sigmoid')(self.embedding_x)
    def forward(self,class_numb):
        dropout_x=self.drop(self.lstm_out)
        dense_x=Dense(50)(dropout_x)
        activation_x=Activation('sigmoid')(dense_x)
        dropout_x=self.drop(activation_x)
        main_loss = Dense(class_numb, activation='sigmoid', name='main_output')(dropout_x)###这里的15是分的类别数
        model = Model(input=self.main_input, output=main_loss)
        return model

def initialize_word_vector():
    n_symbols = len(index_dict) + 1
    embedding_weights = np.zeros((n_symbols, 128))  # 创建一个n_symbols * 128的0矩阵
    for w, index in index_dict.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embedding_weights[index, :] = word_vectors[w]  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）
    return n_symbols,embedding_weights

def train_model(n_symbols,embedding_weights):
    lstm_model_class = LSTM_model(n_symbols,embedding_weights)
    lstm_model = lstm_model_class.forward(classify_numb)
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True) 
    lstm_model.compile(optimizer=sgd,metrics=['accuracy'],loss='categorical_crossentropy')
    print('开始训练模型')
    lstm_model.fit(train_data_emb, train_label_hot, batch_size=batch_size, nb_epoch=n_epoch,validation_data=(test_data_emb, test_label_hot))
    print ("开始验证模型")
    metrics= lstm_model.evaluate(test_data_emb, test_label_hot, batch_size=batch_size, verbose=0)
    for i in range(len(lstm_model.metrics_names)):
        print(str(lstm_model.metrics_names[i]) + ": " + str(metrics[i]))
    lstm_model.save('../class_model/class_model_by_names.h5')  ###自定义的生成模型的路径和模型名称
    export_savedmodel(lstm_model)


if __name__ == '__main__':
    #取数据
    T1 = TextSta(corpus_train_filename)
    allsentences,labels = T1.sen()
    #转换数据为训练数据
    train_data_emb,test_data_emb,train_label_hot,test_label_hot = get_train_data(allsentences,labels)
    #初始化word2vec预训练模型参数
    n_symbols,embedding_weights = initialize_word_vector()
    #加载模型、设置优化器、训练、测试、保存模型
    train_model(n_symbols,embedding_weights)