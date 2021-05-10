#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:09:42 2021

@author: nakaharakan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import f1_score

data_ver=4

train_path='/Users/nakaharakan/Documents/signate_music/process_data/process_data'+str(data_ver)+'/processed_train'+str(data_ver)+'.csv'
test_path='/Users/nakaharakan/Documents/signate_music/process_data/process_data'+str(data_ver)+'/processed_test'+str(data_ver)+'.csv'

def stratified_CV_data(n_fold,train_x,train_y):
    
    skf=SKF(n_splits=n_fold,random_state=None,shuffle=False)
    
    return skf.split(train_x,train_y)

def CV2csv(n_fold):
    
    train_x=pd.read_csv(train_path)
    test_x=pd.read_csv(test_path)
    
    train_y=train_x['genre']
    
    #print(train_x.columns)
    
    del train_x['genre'],train_x['Unnamed: 0'],test_x['Unnamed: 0']
    
    
    
    train_x=np.array(train_x)
    test_x=np.array(test_x)
    train_y=np.array(train_y)
    
    print(train_x.shape)
    
    score=0
    
    
    
    for train_index,val_index in stratified_CV_data(n_fold,train_x,train_y):
        
        tr_x=train_x[train_index]
        tr_y=np.eye(11)[train_y[train_index]]
        
        tr_x=tr_x/np.linalg.norm(tr_x,axis=1).reshape(-1,1)
        
        
        val_x=train_x[val_index]
        val_y=train_y[val_index]
        
        val_x=val_x/np.linalg.norm(val_x,axis=1).reshape(-1,1)
        
        products=np.dot(val_x,tr_x.T)
        
        val_pred=[]
        
        for product in products:
            
            val_pred.append((tr_y*product.reshape(-1,1)).sum(axis=0).argmax())
            
        
            
        score+=f1_score(val_y,np.array(val_pred),average='macro')
        
    print(score/n_fold)
        
    
    
    
    
    
    
    
    
    