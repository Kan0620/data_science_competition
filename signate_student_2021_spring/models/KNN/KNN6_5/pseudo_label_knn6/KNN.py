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

data_ver=6

train_path='/Users/nakaharakan/Documents/signate_music/process_data/process_data'+str(data_ver)+'/processed_train'+str(data_ver)+'.csv'
test_path='/Users/nakaharakan/Documents/signate_music/process_data/process_data'+str(data_ver)+'/processed_test'+str(data_ver)+'.csv'


def stratified_CV_data(n_fold,train_x,train_y):
    
    skf=SKF(n_splits=n_fold,random_state=None,shuffle=False)
    
    return skf.split(train_x,train_y)

def pd_data():
    
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
       'region_unknown']
    
    
    
    train_x[index100]=train_x[index100]*100
    test_x[index100]=test_x[index100]*100
    
    train_x['popularity']=train_x['popularity']*9.5
    test_x['popularity']=test_x['popularity']*9.5
    
    train_x['danceability']=train_x['danceability']*1.5
    test_x['danceability']=test_x['danceability']*1.5
    
    train_x['acousticness']=train_x['acousticness']*1.1
    test_x['acousticness']=test_x['acousticness']*1.1
    
    train_x['positiveness']=train_x['positiveness']*1
    test_x['positiveness']=test_x['positiveness']*1
    
    train_x['duration_ms']=train_x['duration_ms']*1
    test_x['duration_ms']=test_x['duration_ms']*1
    
    
    train_x['fast']=train_x['fast']*0.6
    test_x['fast']=test_x['fast']*0.6
    
    
    train_x['slow']=train_x['slow']*4.2
    test_x['slow']=test_x['slow']*4.2
    
    return train_x,train_y,test_x



def fitter(k=int(),n_fold=int(),pseudo_k=int(),pseudo_n_fold=int()):
    
    
    preds=np.zeros((4046,11))
    
    train_x,train_y,test_x=pd_data()
    
    
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
    
    #===========================予測ここまで==========================
    
    train_x,train_y,test_x=pd_data()
    
    train_x=np.array(train_x)
    train_y=np.array(train_y)
    test_x=np.array(test_x)
    
    test_y=preds
    
    final_preds=np.zeros((4046,11))
    
    score=0
    
    for train_index,test_index in stratified_CV_data(pseudo_n_fold,test_x,test_y):
        
        tr_x=np.concatenate([train_x,test_x[train_index]])
        tr_y=np.concatenate([train_y,test_y[train_index]])
        
        
        val_x=test_x[test_index]
        val_y=test_y[test_index]
        
        print(len(tr_x),len(val_x))
        
        
        knc=KNC(n_neighbors=pseudo_k, weights="distance")
    
        knc.fit(tr_x,tr_y)
        
        pseudo_pred=knc.predict(val_x)
        
        
        
        score+=f1_score(val_y,pseudo_pred,average='macro')
        
        final_preds[test_index]+=np.eye(11)[pseudo_pred]
        
    print(k,pseudo_k,score/pseudo_n_fold)
    
    print('=========='*5)
    
    
    
    sol=pd.DataFrame({'id':[i+4046 for i in range(4046)],'ans':final_preds.argmax(axis=1)})
    
    sol['ans']=sol['ans'].map(lambda x:int(x))
    
    sol.to_csv('/Users/nakaharakan/Documents/signate_music/models/KNN/KNN'+str(data_ver)+'/pseudo_label_knn6/'+str(data_ver)+'KNN_pseudo_label.csv',index=False,header=False)
    
    return final_preds
    
    
def param_decide():
    
    for i in range(8):
        
        fitter(i+1,8,i+1,2)
        
        




    
        






































