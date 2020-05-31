# coding:utf-8
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
from config import *
import logging,os,gc
from data_utils import seg_df,transform_char
from multi_proc_utils import parallelize
from params_utils import w2v_params

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def prepare_w2v_corpus():
    '''
    从网上下载了大众点评语料，解压后近2G,用于训练词向量和字向量
    https://github.com/SophonPlus/ChineseNlpCorpus
    '''
    dianping_df = pd.read_csv(dianping_path)
    comment_df = dianping_df[['comment']]
    comment_df.dropna(inplace=True)
    
    comment_seg_df = parallelize(comment_df,seg_df)
    comment_seg_df['comment'].to_csv(comment_seg_path, header=False, index=None)
    del comment_seg_df
    gc.collect()
    
    comment_df['comment_char'] = comment_df['comment'].apply(transform_char)
    comment_df['comment_char'].to_csv(comment_char_path, header=False, index=None)
    

def train_w2v(data_path,w2v_path,params):
    '''
    训练word2vec
    '''
    logger.info('开始训练word2vec...')
    w2v_model = word2vec.Word2Vec(LineSentence(data_path),
                              size      = params['embed_size'], 
                              min_count = params['min_count'], 
                              window    = params['context_window'],
                              workers   = params['workers'],
                              sg        = params['model'],
                              iter      = params['epochs'])
    # 保存模型
    w2v_model.save(w2v_path)
    logger.info('模型训练完毕并保存')
 
def main():
    '''
    为了后面对比字向量和词向量的效果，这里都训练了。
    词向量的window为5，那么字向量的取10.
    '''
    prepare_w2v_corpus()
    
    params = w2v_params()
    train_w2v(comment_seg_path, w2v_word_path, params)
    
    params['context_window'] = 10
    train_w2v(comment_char_path, w2v_char_path, params)
    

if __name__ == '__main__':
    main()