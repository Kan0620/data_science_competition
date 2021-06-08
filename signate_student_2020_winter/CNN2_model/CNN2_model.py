#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:23:51 2020

@author: nakaharakan
"""

from keras.models import Sequential
from keras.layers.core import Dropout
from keras.layers import Dense,Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D,MaxPooling2D,GlobalMaxPooling2D,Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

def concat(x):#xをテスト
    all_train=pd.read_csv('processed_train3.csv',index_col=0)
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
    
    all_test=pd.read_csv('processed_test3.csv',index_col=0)
    
    
    
    return np.array(all_test)





def CNN2_model(x):
    
    tr_x,tr_y,val_x,val_y=concat(x)
    
    tr_x=tr_x.reshape((-1,12,19,1))
    val_x=val_x.reshape((-1,12,19,1))
    print(7)
    
    

    
    model=Sequential()

    model.add(Conv2D(
    filters=16,
    kernel_size=(1,2),
    input_shape=(12,19,1),
    activation='relu')
         )
    
    model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    input_shape=(12,19,1),
    activation='relu')
         )
    
    model.add(MaxPooling2D((2,2)))
    
    model.add(Conv2D(
    filters=64,
    kernel_size=(2,3),
    input_shape=(12,19,1),
    activation='relu')
         )
    
    
    model.add(MaxPooling2D((2,2)))
    
    
    model.add(Flatten())
    
    
    
    model.add(Dense(
            100,
            activation='relu'))
    
    model.add(Dropout(0.3))
    
    model.add(Dense(
            1,
            activation='sigmoid'))
    
    
    
    model.summary()

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics='acc')

    model.fit(
        tr_x,
        tr_y,
        
        epochs=10,
        validation_data=(val_x,val_y),
        batch_size=32)
    
    return model

def sub_data():
    tes_x=test_x().reshape((-1,12,19,1))
    ans_1=CNN2_model(1).predict(tes_x)
    ans_2=CNN2_model(2).predict(tes_x)
    ans_3=CNN2_model(3).predict(tes_x)
    ans_4=CNN2_model(4).predict(tes_x)
    
    anss=(ans_1+ans_2+ans_3+ans_4)
    
    return anss

def solu(sol,s=str()):
    
    
    sol[np.where(sol>0.5)]=1
    sol[np.where(sol<0.5)]=0
    
    print(sol.shape,sol[:10])
    
    sol=pd.DataFrame({'id':[i+10545 for i in range(10544)],'ans':sol})
    
    sol['ans']=sol['ans'].map(lambda x:int(x))
    
    sol.to_csv('/Users/nakaharakan/Documents/signate/'+s,index=False,header=False)
    
    return sol













