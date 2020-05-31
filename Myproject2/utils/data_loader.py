#coding:utf-8
import pandas as pd
import numpy as np
from config import *
import pickle

def load_pickle(file_path):
    return pickle.load(open(file_path,'rb'))

def build_dataset():
    train_df = pd.read_csv(trainset_proc_path,lineterminator="\n")
    valid_df = pd.read_csv(validset_proc_path,lineterminator="\n")   
    word2id,id2word = load_vocab(mode="word")
    
    train_df['content_seg_pad'] = train_df['content_seg'].apply(lambda x: 
                                                  pad_proc(x, max_len_word, word2id))
    valid_df['content_seg_pad'] = valid_df['content_seg'].apply(lambda x: 
                                                  pad_proc(x, max_len_word, word2id))
    
    
    train_ids_x = train_df['content_seg_pad'].apply(lambda x: transform_data(x, word2id))
    valid_ids_x = valid_df['content_seg_pad'].apply(lambda x: transform_data(x, word2id))

    train_X = np.array(train_ids_x.tolist())
    valid_X = np.array(valid_ids_x.tolist())
 
    np.save(train_x_path, train_X)
    np.save(valid_x_path, valid_X)  

def load_tags():
    '''
    由于分类类别为20类，所以要做20个模型，这一步把20个模型的标签分别拿出来，然后进行one-hot编码
    one hot 编码的维度是4.
    :return:
    '''
    train_df = pd.read_csv(trainset_proc_path,lineterminator="\n")
    valid_df = pd.read_csv(validset_proc_path,lineterminator="\n")

    train_tags = {}
    valid_tags = {}
    for col in tag_columns:
        train_tags[col] = pd.get_dummies(train_df[col])[[-2, -1, 0, 1]].values
        valid_tags[col] = pd.get_dummies(valid_df[col])[[-2, -1, 0, 1]].values 
        
    return train_tags,valid_tags


def load_dataset(max_x_len=200,mode='char'):
    '''
    这一步直接把已经转为id的x和已经进行one hot编码的y，直接准备好，实现数据预处理和训练的解耦。
    还记得吗，进行pad的时候，我们的输入最大长度为350，而这里，
    通过max_x_len这个参数，我们可以自由调整真正训练时候的最大输入长度。
    由于tensorflow 2.0的输入要求是float32格式，所以这里进行一下转化。
    :param max_x_len:
    :return:
    '''
    if mode == 'word':
        train_X = np.load(train_x_path)
        valid_X = np.load(valid_x_path)    
    elif mode == 'char':
        train_X = np.load(train_x_char_path)
        valid_X = np.load(valid_x_char_path)        
        
    train_X = train_X[:,:max_x_len]
    valid_X = valid_X[:,:max_x_len]
    
    train_X = train_X.astype(np.float32)
    valid_X = valid_X.astype(np.float32)        
    train_tags,valid_tags = load_tags()
    
    print("The size of train_X is : {}".format(train_X.shape))
    print("The size of valid_X is : {}".format(valid_X.shape))
    
    print("The size of train_tags: {}".format(train_tags['location_traffic_convenience'].shape))
    print("The size of valid_tags: {}".format(valid_tags['location_traffic_convenience'].shape))    
    
    return train_X, valid_X, train_tags, valid_tags

def load_embedding_matrix(mode='char'):
    '''
    得到训练好的词向量矩阵
    :return:
    '''
    if mode == 'char':
        embedding_matrix = np.load(embedding_matrix_char_path)
    elif mode == 'word':
        embedding_matrix = np.load(embedding_matrix_path)
    return embedding_matrix

def load_vocab(mode="char"):
    '''得到字典'''
    if mode == "char":
        return load_pickle(char2id_path),load_pickle(id2char_path)
    elif mode == "word":
        return load_pickle(word2id_path),load_pickle(id2word_path)
    
def transform_data(sentence,vocab):
    words = sentence.split(' ')
    ids = [vocab[word] if word in vocab else vocab['<UNK>'] for word in words]
    return ids

if __name__ == '__main__':
    build_dataset()
    train_X, valid_X, train_tags,valid_tags = load_dataset(max_x_len=250)
    print("The size of train_X is : {}".format(train_X.shape))
    print("The size of valid_X is : {}".format(valid_X.shape))
    
    print("The size of train_tags: {}".format(train_tags['location_traffic_convenience'].shape))
    print("The size of valid_tags: {}".format(valid_tags['location_traffic_convenience'].shape))
    
    '''
    The size of train_X is : (105000, 250)
    The size of valid_X is : (15000, 250)
    The size of train_tags: (105000, 4)
    The size of valid_tags: (15000, 4)
    '''