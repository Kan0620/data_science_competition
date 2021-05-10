#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 18:53:06 2021

@author: nakaharakan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier as KNC

data_ver=4

train_path='/Users/nakaharakan/Documents/signate_music/process_data/process_data'+str(data_ver)+'/processed_train'+str(data_ver)+'.csv'
test_path='/Users/nakaharakan/Documents/signate_music/process_data/process_data'+str(data_ver)+'/processed_test'+str(data_ver)+'.csv'


def stratified_CV_data(n_fold,train_x,train_y):
    
    skf=SKF(n_splits=n_fold,random_state=None,shuffle=False)
    
    return skf.split(train_x,train_y)



def fitter(k=int(),n_fold=int()):
    
    
    preds=np.zeros((4046,11))
    
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
        tr_y=train_y[train_index]
        
        val_x=train_x[val_index]
        val_y=train_y[val_index]
    
        knc=KNC(n_neighbors=k, weights="distance")
    
        knc.fit(tr_x,tr_y)
        
        pred=knc.predict(val_x)
        
        score+=f1_score(val_y,pred,average='macro')
        
        preds+=np.eye(11)[knc.predict(test_x)]
        
    preds=preds.argmax(axis=1)
        
    print(k,score/n_fold)
    
    sol=pd.DataFrame({'id':[i+4046 for i in range(4046)],'ans':preds})
    
    sol['ans']=sol['ans'].map(lambda x:int(x))
    
    sol.to_csv('/Users/nakaharakan/Documents/signate_music/models/KNN/KNN'+str(data_ver)+'/'+str(data_ver)+'KNN.csv',index=False,header=False)
    
    return sol
    
    
def param_decide():
    
    for i in range(10):
        
        fitter(i+1,10)
        




    
        






































