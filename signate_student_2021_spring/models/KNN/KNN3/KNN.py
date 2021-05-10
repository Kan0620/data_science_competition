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

data_ver=3

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
    
    index100=['region_region_A', 'region_region_B',
       'region_region_C', 'region_region_D', 'region_region_E',
       'region_region_F', 'region_region_G', 'region_region_H',
       'region_region_I', 'region_region_J', 'region_region_K',
       'region_region_L', 'region_region_M', 'region_region_N',
       'region_region_O', 'region_region_P', 'region_region_Q',
       'region_region_R', 'region_region_S', 'region_region_T',
       'region_unknown','nan_acousticness','nan_positiveness','nan_danceability',
              'nan_energy','nan_liveness','nan_speechiness','nan_instrumentalness']
    
    index8='popularity'
    
    train_x[index100]=train_x[index100]*100
    test_x[index100]=test_x[index100]*100
    
    train_x[index8]=train_x[index8]*8
    test_x[index8]=test_x[index8]*8
    
    
    
    
    
    train_x=np.array(train_x)
    test_x=np.array(test_x)
    train_y=np.array(train_y)
    
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
    
    





    
        






































