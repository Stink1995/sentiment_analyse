#coding:utf-8
import argparse

def w2v_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_size", default=200, help="Words embeddings dimension", type=int)
    parser.add_argument("--epochs", default=10, help="train epochs", type=int)
    parser.add_argument("--min_count", default=3, help="min frequence to keep", type=int)
    parser.add_argument("--context_window", default=5, help="context window", type=int)
    parser.add_argument("--workers", default=4, help="number of multiprocessing", type=int)
    parser.add_argument("--model",default=1,help="0 : cbow; 1: skip-gram",type=int) 
    
    args = parser.parse_args()
    params = vars(args)
    return params

def textcnn_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_word_size",default=249687,help="", type=int)
    parser.add_argument("--vocab_char_size",default=12973,help="", type=int)
    parser.add_argument("--batch_size", default=256, help="", type=int)
    parser.add_argument("--max_word_len", default=400, help="", type=int)
    parser.add_argument("--max_char_len", default=800, help="", type=int)
    parser.add_argument("--epochs", default=40, help="", type=int)
    parser.add_argument("--embed_size", default=200, help="", type=int)
    parser.add_argument("--num_filters", default=128, help="", type=int)
    parser.add_argument("--dense_units", default=64, help="", type=int)
    parser.add_argument("--dropout_rate", default=0.5, help="",type=float)
    parser.add_argument("--label_size", default=4, help="",type=int)
    parser.add_argument("--learning_rate", default=1e-4, help="", type=float)
    
    args = parser.parse_args()
    params = vars(args)
    return params

def bilstm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size",default=249687,help="", type=int)
    parser.add_argument("--batch_size", default=256, help="", type=int)
    parser.add_argument("--max_x_len", default=250, help="", type=int)
    parser.add_argument("--epochs", default=20, help="", type=int)
    parser.add_argument("--embed_size", default=200, help="", type=int)
    parser.add_argument("--attn_units", default=64, help="", type=int)
    parser.add_argument("--dropout_rate", default=0.5, help="",type=float)
    parser.add_argument("--label_size", default=4, help="",type=int)
    parser.add_argument("--LSTM_units", default=128, help="", type=int)
    
    args = parser.parse_args()
    params = vars(args)
    return params

if __name__ == '__main__':
    params = w2v_params()
    params['context_window'] = 10
    print()

    