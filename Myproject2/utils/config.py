#coding:utf-8
import os
import pathlib

# 代码的根目录
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

dianping_path = os.path.join(root,'data','raw_data','dianping_comment.csv')
comment_seg_path = os.path.join(root,'data','processed_data','dianping_comment_seg.csv')
comment_char_path = os.path.join(root,'data','processed_data','dianping_comment_char.csv')

# 训练集、验证集和测试集的路径
trainset_path = os.path.join(root,'data','raw_data','sentiment_analysis_trainingset.csv')
validset_path = os.path.join(root,'data','raw_data','sentiment_analysis_validationset.csv')
testset_path = os.path.join(root,'data','raw_data','sentiment_analysis_testa.csv')

# 训练集、验证集和测试集的路径
trainset_proc_path = os.path.join(root,'data','processed_data','sentiment_trainset.csv')
validset_proc_path = os.path.join(root,'data','processed_data','sentiment_validset.csv')

merged_seg_pad_path = os.path.join(root,'data','processed_data','sentiment_merged_seg_pad.csv')
merged_char_pad_path = os.path.join(root,'data','processed_data','sentiment_merged_char_pad.csv')

w2v_word_path = os.path.join(root,'word2vec','w2v_word.model')
w2v_char_path = os.path.join(root,'word2vec','w2v_char.model')

w2v_word_retrain_path = os.path.join(root,'word2vec','w2v_word_retrain.model')
w2v_char_retrain_path = os.path.join(root,'word2vec','w2v_char_retrain.model')

embedding_matrix_path = os.path.join(root, 'word2vec', 'embedding_matrix.npy')
embedding_matrix_char_path = os.path.join(root, 'word2vec', 'embedding_matrix_char.npy')

# 3. numpy 转换后的数据
train_x_path = os.path.join(root, 'data', 'processed_data','train_X.npy')
valid_x_path = os.path.join(root, 'data', 'processed_data','valid_X.npy')

train_x_char_path = os.path.join(root, 'data', 'processed_data','train_X_char.npy')
valid_x_char_path = os.path.join(root, 'data', 'processed_data','valid_X_char.npy')

# 为计算句子向量，提前准备好的4个文件
word2id_path = os.path.join(root,'data','processed_data','word2id.pickle')
id2word_path = os.path.join(root,'data','processed_data','id2word.pickle')

char2id_path = os.path.join(root,'data','processed_data','char2id.pickle')
id2char_path = os.path.join(root,'data','processed_data','id2char.pickle')

tags_path = os.path.join(root,'data','processed_data','tags.txt')
tag_columns = [tag.strip() for tag in open(tags_path).readlines()]

textcnn_dir = os.path.join(root,'data','textcnn_results')
bilstm_dir = os.path.join(root,'data','bilstm_results')

max_len_word = 400
max_len_char = 800
vocab_word_size = 249687
vacab_char_size = 12973

