#coding:utf-8
import sys,os
sys.path.append("../utils")
import tensorflow as tf
from textcnn_model import TextCNN
from config import *
from gpu_utils import config_gpu_one
from params_utils import textcnn_params
from textcnn_train_helper import train_model
from tensorflow.keras.backend import clear_session
import gc
from data_loader import load_embedding_matrix,load_dataset
from batcher import batch_generator
import datetime
from sklearn.metrics import f1_score
import numpy as np

# train_model函数里面有@tf.function装饰器，可能导致不能调试，所以加这一句
tf.config.experimental_run_functions_eagerly(True)
# 设置gpu，显存根据需要而分配，而不是占满显存。
config_gpu_one()

params = textcnn_params()
mode = "word"
embed_matrix = load_embedding_matrix(mode)
# 加载已经转化为词的id的数据集，标签已经转化为了one hot编码。
train_X, valid_X,train_tags,valid_tags = load_dataset(params["max_word_len"],mode)
date = datetime.datetime.now().date().strftime('%Y-%m-%d').replace('-','')

# 计算每个小类下4个类别的频率，进而得到类别的权重，用于自定义loss函数，缓解类别不平衡问题（代价敏感学习）
def class_count(df):
    df_label = np.argmax(df,1)
    class_freqs = 1 / np.bincount(df_label) 
    class_freqs = class_freqs / (max(class_freqs) - min(class_freqs))
    return tf.convert_to_tensor(class_freqs,dtype=tf.float32)

F1_score = 0
for tag in tag_columns:
    
    train_Y = train_tags[tag]
    valid_Y = valid_tags[tag]
    weights = class_count(train_Y)
    
    print("Building the model for catetory: {}".format(tag))
    model = TextCNN(params, embed_matrix)    
    
    print("Starting the training model for catetory: {}".format(tag))
    save_path = os.path.join(textcnn_dir,date,mode,tag)
    if os.path.exists(save_path):
        print('dir exists') 
    else:
        print('dir not exists, create dir.')
        os.makedirs(save_path)
    
    train_data, train_steps = batch_generator(train_X, train_Y, params["batch_size"])
    valid_data, valid_steps = batch_generator(valid_X, valid_Y, params["batch_size"])
    
    train_model(model, 
                train_data,
                valid_data,
                train_steps,
                valid_steps,
                params,
                weights,
                save_path)
    
    print("Finish the training  for catetory: {}".format(tag))
    # 把模型删除，重置，用于训练下一个模型.
    
    del model
    gc.collect()
    clear_session()

    # 直接进行模型评估，同时测试模型加载预测是否正常。
    model = tf.saved_model.load(save_path)
    y_list,pred_list = [],[]
    for x, y in valid_data.take(valid_steps):
        y_pred = model.call(x)
        y,y_pred = tf.argmax(y,1).numpy(),tf.argmax(y_pred,1).numpy()
        y_list.append(y)
        pred_list.append(y_pred)
        
    f1  = f1_score(tf.concat(y_list, 0), tf.concat(pred_list, 0), average='macro')
    print("F1-score of model for catetory {} is {:.4f}".format(tag, f1))

    F1_score += f1

print("F1-score of the sentiment analysis model is {}".format(F1_score / 20))