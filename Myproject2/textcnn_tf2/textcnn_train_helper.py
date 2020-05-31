#coding:utf-8
import sys
sys.path.append('../utils')
import tensorflow as tf
import time
from data_utils import *
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
    
def train_model(model, 
                train_data,
                valid_data,
                train_steps,
                valid_steps,
                params, 
                weights,
                save_path):
    
    epochs = params['epochs']
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,clipnorm=2.0,decay=1e-7)

    # 保存训练过程的loss，用于可视化。
    train_loss_results = []
    valid_loss_results = []
    F1_score_results = []

    # 自定义loss函数，用代价敏感学习来缓解类别不平衡问题。
    def loss_function(real,pred,weights):
        batch_loss = - tf.reduce_sum(weights * real * tf.math.log(pred)) 
        return batch_loss / len(real)
    
    def evaluate(valid_data,valid_steps):
        y_list,pred_list = [],[]
        valid_loss = 0
        for x,y in valid_data.take(valid_steps):
            y,y_pred,batch_loss = valid_step(x,y)
            y_list.append(y)
            pred_list.append(y_pred)
            valid_loss += batch_loss

        F1_score = f1_score(tf.concat(y_list, 0), tf.concat(pred_list, 0), average='macro')
        return valid_loss / valid_steps,F1_score
    
    @tf.function
    def valid_step(x,y):
        y_pred = model(x)
        batch_loss = loss_function(y,y_pred,weights)
        y,y_pred = tf.argmax(y,1).numpy(),tf.argmax(y_pred,1).numpy()
        return y,y_pred,batch_loss

    # 增加这个装饰器，可以加速计算。
    @tf.function                            
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x)
            batch_loss = loss_function(y, y_pred,weights)
        variables = model.trainable_variables
        gradients = tape.gradient(batch_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    # 用early stop 防止过拟合。如果超过500个batch还没提升，就停止训练。
    best_F1_score = 0
    best_step = 0
    break_flag = False
      
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        for (batch, (x, y)) in enumerate(train_data.take(train_steps)):
            batch_loss = train_step(x, y)
            total_loss += batch_loss

            step = optimizer.iterations.numpy()
            # 每40个batch评估一次模型，如果F1-score 有提升就保存模型
            if (batch+1) % 40 == 0:
                valid_loss,F1_score = evaluate(valid_data,valid_steps)
                
                template = 'Epoch {} | Step {} | Batch {} | Train Loss {:.4f} | Valid Loss {:.4f} | Valid F1-score {:.4f}'
                print(template.format(epoch + 1, step, batch+1 , batch_loss, valid_loss,F1_score))
                
                if F1_score > best_F1_score:
                    best_F1_score = F1_score
                    best_step = step
                    # 不用Checkpoint的方式保存，为了方便部署。可是保存整个模型，而不是参数，导致模型很大，每个模型200M，20个模型4G.
                    tf.saved_model.save(model,save_path)
                    print('Saving model at epoch {} step {} in {}'.format(epoch + 1,
                                                                          step,
                                                                          save_path))
            # 如果超过2000个batch了，还没有提升，那就停止训练。大概是5个epoch。
            if (step - best_step) >= 2000:
                break_flag = True
                break
        
        # 每个epoch结束了，再次评估模型
        valid_loss,F1_score = evaluate(valid_data,valid_steps)
        valid_loss_results.append(valid_loss)
        F1_score_results.append(F1_score)
        train_loss = total_loss / train_steps
        train_loss_results.append(train_loss)

        template = 'Epoch {} | Train Loss {:.4f} | Valid Loss {:.4f} | Valid F1-Score {:.4f}'
        print(template.format(epoch + 1,train_loss, valid_loss, F1_score))
        print('Time taken for 1 epoch {}'.format(get_time_dif(start)))         
                
        if break_flag:
            print("No optimization for a lont time, auto-stopping... ")
            print("Best F1-score is {:.4f},at step {}".format(best_F1_score,best_step))
            break
    
    # 训练和验证的loss，F1值，进行可视化
    epochs = list(range(len(train_loss_results)))  
    plt.plot(epochs, train_loss_results, 'bo',label='Traning loss')
    plt.plot(epochs, valid_loss_results, 'b',label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path+'_loss.png')
    plt.clf()

    plt.plot(epochs, F1_score_results, 'b', label='Validation F1-score')
    plt.title('Validation F1-score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-score')
    plt.savefig(save_path + '_F1-score.png')
    plt.clf()