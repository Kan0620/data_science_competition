#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:23:51 2020

@author: nakaharakan
"""

from keras.models import Sequential,load_model
from keras.layers.core import Dropout
from keras.layers import Dense,Activation,Reshape,Conv1D,MaxPooling1D,Flatten,BatchNormalization
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np

def concat(x):#xをテスト
    all_train=pd.read_csv('processed_train10.csv',index_col=0)
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
    
    all_test=pd.read_csv('processed_test10.csv',index_col=0)
    
    
    
    return np.array(all_test)


def Encoder():
    
    tr_x,tr_y,val_x,val_y=concat(1)
    testx=test_x()
    
    enc_x=np.concatenate([tr_x,val_x,testx],0)
    
    model=Sequential()

    model.add(Dense(
        200,
        input_dim=tr_x.shape[1],

        ))    
    
    model.add(Activation('relu'))
    
    model.add(Dense(150))
    
    model.add(Activation('relu'))
    
    model.add(Dense(
        200,
        input_dim=tr_x.shape[1],

        ))
    
    model.add(Activation('relu'))
    
    model.add(Dense(tr_x.shape[1]))
    
    model.compile(
        loss='mse',
        optimizer='adam',
        )
    
    reduce_lr=ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.8,
            patience=6,
            verbose=1,
            mode='max',
            min_lr=0.0001
            )
    
    model.fit(
        enc_x,enc_x,
        epochs=50,
        validation_data=(val_x,val_x),
        batch_size=32,
        callbacks=[reduce_lr]
        )
    
    return model




def NN_model(x):
    
    
    tr_x,tr_y,val_x,val_y=concat(x)
    
    print(tr_x.shape,val_x.shape)

    model=Sequential()

    model.add(Dense(
        200,
        input_dim=tr_x.shape[1],

        ))    
    
    
        
    model.add(Activation('relu'))
    
    
    model.add(Dropout(0.5))
    
    model.add(Dense(100))
    
    model.add(BatchNormalization())
    
    model.add(Activation('relu'))
    
    model.add(Dropout(0.5))

    model.add(Dense(
        1,
        
        ))     

    model.add(Activation('sigmoid'))    

    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics='acc')
    model.summary()
    
    
    reduce_lr=ReduceLROnPlateau(
            monitor='val_acc',
            factor=0.8,
            patience=5,
            verbose=1,
            mode='max',
            min_lr=0.0001
            )

    model.fit(
        tr_x,tr_y,
        epochs=15,
        validation_data=(val_x,val_y),
        batch_size=32,
        callbacks=[]
        )
    
    return model

def sub_data():
    model_1=NN_model(1)
    ans_1=model_1.predict(test_x())
    ans_2=NN_model(2).predict(test_x())
    ans_3=NN_model(3).predict(test_x())
    ans_4=NN_model(4).predict(test_x())
    model_1.summary()
    
    anss=(ans_1+ans_2+ans_3+ans_4)
    
    
    
    return anss.reshape((-1))


def solu(sol,s=str()):
    #0.4956851
    
    sol[np.where(sol>0.4956851)]=1
    sol[np.where(sol<0.4956851)]=0
    
    print(sol.shape,sol[:10])
    
    sol=pd.DataFrame({'id':[i+10545 for i in range(10544)],'ans':sol})
    
    sol['ans']=sol['ans'].map(lambda x:int(x))
    
    sol.to_csv('/Users/nakaharakan/Documents/signate/'+s,index=False,header=False)
    
    return sol












