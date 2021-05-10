#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 01:12:57 2021

@author: nakaharakan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import f1_score
import xgboost as xgb
import glob
from sklearn.metrics import f1_score


        
        
        



def stratified_CV_data(n_fold,train_x,train_y):
    
    skf=SKF(n_splits=n_fold,random_state=None,shuffle=False)
    
    return skf.split(train_x,train_y)


def to_csv(weight_array):
    
    train_paths=['9GBDT_train_x.csv','9random_forest_train_x.csv']
    test_paths=['9GBDT_test_x.csv','9random_forest_test_x.csv']
    
    train_path='/Users/nakaharakan/Documents/signate_music/train.csv'
    
    tr_y=np.array(pd.read_csv(train_path)['genre'])
    
    train_ans=np.zeros((4046,11))
    test_ans=np.zeros((4046,11))
    
    for weighter,train_path,test_path in zip(weight_array,train_paths,test_paths):
        
        print(weighter)
        
        train_ans+=weighter*np.array(pd.read_csv(train_path))
        
        test_ans+=weighter*np.array(pd.read_csv(test_path))
        
    
        
    print(f1_score(tr_y,train_ans.argmax(axis=1),average='macro'))
    
    sol=pd.DataFrame({'id':[i+4046 for i in range(4046)],'ans':test_ans.argmax(axis=1)})
    
    sol['ans']=sol['ans'].map(lambda x:int(x))
    
    sol.to_csv('/Users/nakaharakan/Documents/signate_music/ensemble/layer0/weighted_average.csv',index=False,header=False)
    
        
    
        
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        