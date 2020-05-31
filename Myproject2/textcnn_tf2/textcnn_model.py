#coding:utf-8
import sys
sys.path.append('../utils')
from config import *
from data_loader import load_embedding_matrix
import tensorflow as tf
from tensorflow.keras import layers
from params_utils import textcnn_params
from gpu_utils import config_gpu_one


class TextCNN(tf.keras.Model):
    def __init__(self,params,embed_matrix):
        super(TextCNN,self).__init__()
        self.params = params
        self.embedding = layers.Embedding(params["vocab_word_size"],
                                          params["embed_size"],
                                          weights=[embed_matrix],
                                          trainable=False)
        # filter sizes are 2,3,4,5
        self.convs = [layers.Conv1D(params["num_filters"], 
                                    kernel_size=fsz,
                                    padding="same",
                                    activation="relu") for fsz in [2,3,4,5]]
        
        self.pool = layers.GlobalMaxPooling1D()
        self.flatten = layers.Flatten()
        self.concat = layers.Concatenate()
        self.dropout = layers.Dropout(params["dropout_rate"])
        self.fc = layers.Dense(params["dense_units"],activation="relu")
        self.linear = layers.Dense(params["label_size"])
    
    ''''
    使用继承 tf.keras.Model 类建立的 Keras 模型以SavedModel的方式保存，
    须注意 call 方法需要以 @tf.function 修饰，以转化为 SavedModel 支持的计算图
    模型导入须注意模型推断时需要显式调用 call 方法
    必须在里面传入参数，不然训练好的模型，没法用 model.call(x)来预测
    '''
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None),dtype=tf.float32)])
    def call(self,inp):
        emb_inp = self.embedding(inp)
        convs = []
        for conv_layer in self.convs:
            conv_inp = conv_layer(emb_inp)
            pool_inp = self.pool(conv_inp)
            flat_inp = self.flatten(pool_inp)
            convs.append(flat_inp)
        concat_inp = self.concat(convs)
        out = self.dropout(concat_inp)
        out = self.fc(out)
        out = self.linear(out)
        return tf.nn.softmax(out,axis=1)
        
       
if __name__ == "__main__":
    config_gpu_one()
    params = textcnn_params()
    mode = "word"
    
    embed_matrix = load_embedding_matrix(mode)
    model = TextCNN(params, embed_matrix)
    
    example_input = tf.ones(shape=(params["batch_size"],
                                   params["max_word_len"]),
                            dtype=tf.float32)
    example_output = model(example_input)
    print('Textcnn output shape: (batch size,label size) {}'.format(example_output.shape))