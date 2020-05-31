# coding:utf-8
import sys
sys.path.append('../utils')
from config import *
from data_loader import load_embedding_matrix
import tensorflow as tf
from tensorflow.keras import layers
from params_utils import textcnn_params, bilstm_params
from gpu_utils import config_gpu_one
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints


class Attention1(tf.keras.layers.Layer):
    def __init__(self, attention_size, **kwargs):
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        et = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class Attention2(tf.keras.layers.Layer):
    def __init__(self, attn_units):
        super(Attention, self).__init__()
        self.W = tf.keras.layers.Dense(attn_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x):
        et = self.V(tf.nn.tanh(self.W(x)))
        attn_weights = tf.nn.softmax(tf.squeeze(et, axis=-1))
        attn_weights = tf.expand_dims(attn_weights, axis=-1)
        out = attn_weights * x
        return tf.reduce_sum(out, axis=1)

class TextBilstmAtt(tf.keras.Model):
    def __init__(self, params, embed_matrix):
        super(TextBilstmAtt, self).__init__()
        self.embedding = layers.Embedding(params["vocab_size"],
                                          params["embed_size"],
                                          weights=[embed_matrix],
                                          trainable=False)
        self.bilstm = layers.Bidirectional(layers.LSTM(params["LSTM_units"],
                                                       dropout=params["dropout_rate"],
                                                       return_sequences=True))
        self.attention = Attention(params["attn_units"])
        self.linear = layers.Dense(params["label_size"])

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32)])
    def call(self, inp):
        emb_inp = self.embedding(inp)
        out = self.bilstm(emb_inp)
        out = self.attention(out)
        return self.linear(out)


if __name__ == "__main__":
    config_gpu_one()
    params = bilstm_params()

    embed_matrix = load_embedding_matrix()
    Bilstm_Attn = TextBilstmAtt(params, embed_matrix)

    example_input = tf.ones(shape=(params["batch_size"],
                                   params["max_x_len"]),
                            dtype=tf.int32)
    example_output = Bilstm_Attn(example_input)
    print('Textcnn output shape: (batch size,label size) {}'.format(example_output.shape))