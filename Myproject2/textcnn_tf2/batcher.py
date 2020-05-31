#coding:utf-8
import sys
sys.path.append('../utils')
from data_loader import load_dataset
import tensorflow as tf
import numpy as np
import math
from config import tag_columns

def batch_generator(train_X,train_Y,batch_size,sample_num=None):
    '''
    这一步用tf.data生成batch，非常常用的操作。
    ：sample_num: 一开始要把模型调通，那么只需要取小量数据，比如10000.
    '''
    if sample_num:
        train_X = train_X[:sample_num]
        train_Y = train_Y[:sample_num]
    train_X = tf.cast(train_X,dtype=tf.float32)
    train_Y = tf.cast(train_Y,dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((train_X,train_Y)).shuffle(len(train_X))
    # 如果不足一个batch，那就扔掉
    dataset = dataset.batch(batch_size, drop_remainder=True)
    steps_per_epoch = len(train_X) // batch_size
    return dataset, steps_per_epoch

def test_batch_generator(test_X,batch_size):
    '''
    这个函数暂时用不到，这是比赛时，用来预测和提交结果时，用的。
    :param test_X:
    :param batch_size:
    :return:
    '''
    dataset = tf.data.Dataset.from_tensor_slices(test_X)
    # 比赛的话，预测时，不足一个batch的，不能扔掉。
    dataset = dataset.batch(batch_size)
    steps_per_epoch = math.ceil(len(test_X)/batch_size)
    return dataset,steps_per_epoch


if __name__ == '__main__':
    
    train_X, valid_X, train_tags,valid_tags = load_dataset(max_x_len=200)
    for tag in tag_columns:
        train_Y = train_tags[tag]
        dataset, steps_per_epoch = batch_generator(train_X,train_Y,batch_size=64)
        for (batch, (inputs, targets)) in enumerate(dataset.take(steps_per_epoch)): 
            print("The size of inputs :{}".format(inputs.shape))
            print("The size of targets :{}".format(targets.shape))
    