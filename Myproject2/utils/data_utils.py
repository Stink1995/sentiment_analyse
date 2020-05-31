#coding:utf-8
import numpy as np
import pickle,json,jieba,re
from gensim.models import word2vec
from config import *
from collections import Counter
from itertools import chain
import pandas as pd
import time
from datetime import timedelta
import zhconv

def save_pickle(s,file_path):
    pickle.dump(s,open(file_path,'wb'))

def load_pickle(file_path):
    return pickle.load(open(file_path,'rb'))

def get_time_dif(start_time):
    '''
    :param start_time: 开始时间
    :return:  耗费的时间 00:00:00
    '''
    end_time = time.time()
    time_dif = end_time - start_time 
    return timedelta(seconds=int(round(time_dif)))  


def clean_sentence(sentence):
    '''
    特殊符号去除,汉语标点符号就不去掉了
    :param sentence: 待处理的字符串
    :return: 过滤特殊字符后的字符串
    '''
    if isinstance(sentence, str):
        return re.sub(
            r'[\|\/\[\]\{\}_^*(+\"\')\n]+|[；，。:：+——(),%【】“”@#￥……&*（）]+',
            '', sentence)
    else:
        return ''

def sentence_seg(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    # sentence = clean_sentence(sentence)
    words = jieba.lcut(sentence)
    return ' '.join(words)

def seg_df(df):
    '''
    :param df: pd.DataFrame格式的数据
    :return: 分词后的数据
    '''
    for col in df.columns.tolist():
        df[col] = df[col].apply(sentence_seg)
    return df

def load_word2vec(word2vec_path):
    '''
    ：return： w2v模型，所有词和对应的embedding
    '''
    model = word2vec.Word2Vec.load(word2vec_path)
    vocab = model.wv.index2word
    embeddings = model.wv.vectors
    return model,vocab,embeddings

def build_vocab(vocab):
    word2id = {word: index for index, word in enumerate(vocab)}
    id2word= {index: word for index, word in enumerate(vocab)}
    return word2id,id2word

def pad_proc(sentence,max_len,vocab):
    words = sentence.strip().split(' ')
    words = words[:max_len]
    sentence = [word if word in vocab else '<UNK>'for word in words]
    sentence = sentence + ['<PAD>'] * (max_len - len(words))
    return ' '.join(sentence)


def transform_data(sentence,vocab):
    words = sentence.split(' ')
    ids = [vocab[word] if word in vocab else vocab['<UNK>'] for word in words]
    return ids

def transform_char(sentence):
    # sentence = clean_sentence(sentence)
    return ' '.join(list(sentence.strip()))


def traditional_2_simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = clean_sentence(sentence)
    return zhconv.convert(sentence, 'zh-cn')

if __name__ == '__main__':
    # get_max_len(data)
    # w2v_model, _, _ =load_word2vec()
    # print(w2v_model.wv.most_similar(['说'],topn=10))
    # embedding_matrix = load_embedding_matrix()
    # print()
    word2id,id2word = load_vocab()