#coding:utf-8
import sys
sys.path.append('../utils')
from data_loader import load_vocab
import tensorflow as tf
import numpy as np
import math,os,re
from config import *
import pandas as pd
import datetime
from gpu_utils import config_gpu_one
import jieba

class Vocab:

    def __init__(self,mode):
        """
        Vocab 对象,vocab基本操作封装
        用于预测时，对于每一条评论，转化为id等操作
        """
        self.word2id, self.id2word = load_vocab(mode)
        self.count = len(self.word2id)

    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id["<UNK>"]
        return self.word2id[word]

    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def size(self):
        return self.count

def clean_sentence(sentence):
    '''
    特殊符号去除,汉语标点符号就不去掉了
    :param sentence: 待处理的字符串
    :return: 过滤特殊字符后的字符串
    '''
    if isinstance(sentence, str):
        return re.sub(
            r'[\|\/\[\]\{\}_^*(+\"\')\n]+|[；，。.:：+——(),%【】“”@#￥……&*（）]+',
            '', sentence)
    else:
        return ''

def transform_char(sentence):
    '''
    把句子拆分为字，为字向量模型做准备
    '''
    sentence = clean_sentence(sentence)
    return  [char for char in sentence.strip() if char.strip()]

def sentence_seg(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    sentence = clean_sentence(sentence)
    return [word for word in jieba.lcut(sentence) if word.strip()]


def transform_class(arr):
    '''
    把模型预测的结果转化为数据集中的标记[-2,-1,0,1]
    '''
    arr = list(arr)
    if arr.index(max(arr)) == 0:
        return -2
    elif arr.index(max(arr)) == 1:
        return -1
    elif arr.index(max(arr)) == 2:
        return 0
    else:
        return 1

class ModelPredict:
    def __init__(self,mode,max_len,model_dir):
        self.models = self.load_model(mode,model_dir)
        self.vocab = Vocab(mode)
        self.max_len = max_len
     
    """加载20个模型"""   
    def load_model(self,mode,model_dir):
        models = []
        for index,tag in enumerate(tag_columns):
            save_path = os.path.join(model_dir,tag)
            model = tf.saved_model.load(save_path)
            models.append(model)
        return models
    
    """把评论转化为id，并进行pad"""
    def transfer_id(self, article, vocab):
        article = sentence_seg(article)
        words = article[: self.max_len]
        ids = [self.vocab.word_to_id(w) for w in words]
        ids += [self.vocab.word_to_id("<PAD>")] * (self.max_len - len(words))
        return ids
    
    """进行预测"""
    def predict(self,article):
        input_ids = self.transfer_id(article,self.vocab)
        input_ids = tf.convert_to_tensor([input_ids],dtype=tf.float32)
        pred_labels = {}
        for tag,model in zip(tag_columns,self.models):
            y_pred = model.call(input_ids)
            label = transform_class(y_pred.numpy()[0])
            pred_labels[tag] = label
        return pred_labels
       

if __name__ == "__main__":
    config_gpu_one()
    
    content = open("predict_comment.txt",encoding="utf-8").readlines()[0]
    
    mode = "word"
    
    """ 部署的时候，把这个日期改一下，改成自己训练模型的日期"""
    date = "20200107"
    save_dir = os.path.join(textcnn_dir,date,mode)
    
    max_len_word = 400
    model = ModelPredict(mode, max_len_word, save_dir)
    y_pred = model.predict(content)
    print(y_pred)
    
    """ {'location_traffic_convenience': -2, 
         'location_distance_from_business_district': -2, 
         ...,
         'others_willing_to_consume_again': 1}"""

