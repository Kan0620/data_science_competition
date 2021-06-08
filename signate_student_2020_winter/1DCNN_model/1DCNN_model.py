#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 15:41:08 2021

@author: nakaharakan
"""

from keras.models import Sequential
from keras.layers.core import Dropout
from keras.layers import Dense,Activation,Reshape,Conv1D,Conv2D,MaxPooling1D,Flatten,BatchNormalization
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import GRU
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np

def concat(x):#xをテスト
    all_train=pd.read_csv('processed_train.csv',index_col=0)
    data1=all_train[:2600]
    data2=all_train[2600:5200]
    data3=all_train[5200:7800]
    data4=all_train[7800:]

    all_train=[data1,data2,data3,data4]
    
    val_data=all_train.pop(x-1)
    train_data=pd.concat((all_train[0],all_train[1],all_train[2]))
    
    
    tr_y=train_data['state']
    val_y=val_data['state']
    
    
    
    
    del train_data['state']
    del val_data['state']
    
    
    
    return np.array(train_data),np.array(tr_y),np.array(val_data),np.array(val_y)

def test_x():
    
    all_test=pd.read_csv('processed_test.csv',index_col=0)
    
    
    
    return np.array(all_test)







def DCNN_model(x):
    
    
    
    tr_x,tr_y,val_x,val_y=concat(x)
    
    print(tr_x.shape,val_x.shape)

    model=Sequential()

    model.add(Dense(
        200,
        input_dim=tr_x.shape[1],

        ))    
    
    
        
    model.add(Activation('relu'))
    
    model.add(Dropout(0.5))
    
    model.add(Reshape((10,20,1)))
    
    model.add(Conv2D(
            filters=8,
            kernel_size=(10,1),
            strides=1,
            
            activation='relu'))
    
    model.add(Reshape((20,8)))
    
    model.add(GRU(20
,
input_shape=(20,8),
))
    
    
    
    
    model.add(Dropout(0.5))
    
    

    model.add(Dense(
        1,
        
        ))     

    model.add(Activation('sigmoid'))    

    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics='acc')
    
    reduce_lr=ReduceLROnPlateau(
            monitor='val_acc',
            factor=0.8,
            patience=3,
            verbose=1,
            mode='max',
            min_lr=0.0001
            )
    
    model.summary()

    model.fit(
        tr_x,tr_y,
        epochs=50,
        validation_data=(val_x,val_y),
        batch_size=32,
        callbacks=[reduce_lr]
        )
    
    return model
    
    
    

def sub_data():
    model_1=DCNN_model(1)
    ans_1=model_1.predict(test_x())
    ans_2=DCNN_model(2).predict(test_x())
    ans_3=DCNN_model(3).predict(test_x())
    ans_4=DCNN_model(4).predict(test_x())
    model_1.summary()
    
    anss=(ans_1+ans_2+ans_3+ans_4)
    
    
    
    return anss.reshape((-1))


def solu(sol,s=str()):
    
    
    sol[np.where(sol>0.5)]=1
    sol[np.where(sol<0.5)]=0
    
    print(sol.shape,sol[:10])
    
    sol=pd.DataFrame({'id':[i+10545 for i in range(10544)],'ans':sol})
    
    sol['ans']=sol['ans'].map(lambda x:int(x))
    
    sol.to_csv('/Users/nakaharakan/Documents/signate/'+s,index=False,header=False)
    
    return sol