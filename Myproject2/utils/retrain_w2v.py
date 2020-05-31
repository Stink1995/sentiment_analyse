# coding:utf-8
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
from config import *
import logging,os

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def retrain_w2v():
    '''
    首先对输入的内容进行处理：一是把不在字典中词替换为<UNK>,
    二是如果不足最大输入长度，那就进行PAD，加上<PAD>，三是超过最大长度，就截断。
    然后增量训练word2vec，就可以得到<PAD>和<UNK>的字向量！
    训练好word2vec后，重新建立word2id的字典，同时把输入内容由词转化为id，保存备用，后面我们就可以直接输入模型啦！！
    :return:
    '''
    train_df = pd.read_csv(trainset_proc_path,lineterminator="\n")
    valid_df = pd.read_csv(validset_proc_path,lineterminator="\n")    
    w2v_model,vocab,_ = load_word2vec(w2v_word_path)
    word2id,id2word = build_vocab(vocab)
    
    train_df['content_seg_pad'] = train_df['content_seg'].apply(lambda x: 
                                                  pad_proc(x, max_len_word, word2id))
    valid_df['content_seg_pad'] = valid_df['content_seg'].apply(lambda x: 
                                                  pad_proc(x, max_len_word, word2id))
    merged_seg_pad_df = pd.concat([train_df[['content_seg_pad']],valid_df[['content_seg_pad']]],axis=0)
    merged_seg_pad_df.to_csv(merged_seg_pad_path,header=False,index=None)
    
    print('start retrain w2v model')
    w2v_model.build_vocab(LineSentence(merged_seg_pad_path),update=True)
    w2v_model.train(LineSentence(merged_seg_pad_path),epochs=5,total_examples=w2v_model.corpus_count)
    w2v_model.save(w2v_word_retrain_path)
    print('finish retrain w2v model')
    print('final w2v_model has vocabulary of ', len(w2v_model.wv.vocab))
    
    vocab = w2v_model.wv.index2word
    embedding_matrix = w2v_model.wv.vectors
    word2id,id2word = build_vocab(vocab)
    
    save_pickle(word2id,word2id_path)
    save_pickle(id2word,id2word_path)
    np.save(embedding_matrix_path, embedding_matrix)
    
    train_ids_x = train_df['content_seg_pad'].apply(lambda x: transform_data(x, word2id))
    valid_ids_x = valid_df['content_seg_pad'].apply(lambda x: transform_data(x, word2id))

    train_X = np.array(train_ids_x.tolist())
    valid_X = np.array(valid_ids_x.tolist())
 
    np.save(train_x_path, train_X)
    np.save(valid_x_path, valid_X)  


def retrain_char2vec():
    train_df = pd.read_csv(trainset_proc_path,lineterminator="\n")
    valid_df = pd.read_csv(validset_proc_path,lineterminator="\n")    
    char2vec_model,vocab,_ = load_word2vec(w2v_char_path)
    char2id,id2char = build_vocab(vocab)
    
    train_df['content_char_pad'] = train_df['content_char'].apply(lambda x: 
                                                  pad_proc(x, max_len_char, char2id))
    valid_df['content_char_pad'] = valid_df['content_char'].apply(lambda x: 
                                                  pad_proc(x, max_len_char, char2id))
    merged_char_pad_df = pd.concat([train_df[['content_char_pad']],valid_df[['content_char_pad']]],axis=0)
    merged_char_pad_df.to_csv(merged_char_pad_path,header=False,index=None)
    
    print('start retrain char2vec model')
    char2vec_model.build_vocab(LineSentence(merged_char_pad_path),update=True)
    char2vec_model.train(LineSentence(merged_char_pad_path),epochs=5,total_examples=char2vec_model.corpus_count)
    char2vec_model.save(w2v_char_retrain_path)
    print('finish retrain char2vec model')
    print('final char2vec model has vocabulary of ', len(char2vec_model.wv.vocab))
    
    vocab = char2vec_model.wv.index2word
    embedding_matrix = char2vec_model.wv.vectors
    char2id,id2char = build_vocab(vocab) 
    
    save_pickle(char2id,char2id_path)
    save_pickle(id2char,id2char_path)
    np.save(embedding_matrix_char_path, embedding_matrix)
    
    train_ids_x = train_df['content_char_pad'].apply(lambda x: transform_data(x, char2id))
    valid_ids_x = valid_df['content_char_pad'].apply(lambda x: transform_data(x, char2id))

    train_X = np.array(train_ids_x.tolist())
    valid_X = np.array(valid_ids_x.tolist())
 
    np.save(train_x_char_path, train_X)
    np.save(valid_x_char_path, valid_X)    

def load_word2vec(word2vec_path):
    '''
    以model二进制文件的形式加载，占用内存大大减少
    embedding以np.array的格式保存，也极大减少了内存占用
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

if __name__ == '__main__':
    retrain_w2v()
    # retrain_char2vec()