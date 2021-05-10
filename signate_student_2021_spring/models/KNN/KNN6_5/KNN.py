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

train_raw_pop=np.array(pd.read_csv('/Users/nakaharakan/Documents/signate_music/train.csv')['popularity'])
test_raw_pop=np.array(pd.read_csv('/Users/nakaharakan/Documents/signate_music/test.csv')['popularity'])



for i in range(7):
    
    
    print(i,len(np.where((i*12<train_raw_pop)*(train_raw_pop<(i+1)*12))[0]))

def stratified_CV_data(n_fold,train_x,train_y):
    
    skf=SKF(n_splits=n_fold,random_state=None,shuffle=False)
    
    return skf.split(train_x,train_y)



def fitter(k=int(),n_fold=int(),target_fold=int(),param=float()):
    
    out_train_y=np.zeros((4046,11))
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
       'region_unknown']
    
    
    
    train_x[index100]=train_x[index100]*100
    test_x[index100]=test_x[index100]*100
    
    train_x['popularity']=train_x['popularity']*7
    test_x['popularity']=test_x['popularity']*7
    
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
    
    
    
    train_x=np.array(train_x)
    test_x=np.array(test_x)
    train_y=np.array(train_y)
    
    
    
    score=0
    
    
    for train_index,val_index in stratified_CV_data(n_fold,train_x,train_y):
        
        tr_x=train_x[train_index]
        tr_y=train_y[train_index]
        
        val_x=train_x[val_index]
        val_y=train_y[val_index]
        
        tr_pop=train_raw_pop[train_index]
        val_pop=train_raw_pop[val_index]
        
        tr_target_encode=np.zeros((len(tr_x),11))
        val_target_encode=np.zeros((len(val_x),11))
        test_target_encode=np.zeros((len(test_x),11))
        
        
        for label_index,labeled_index in stratified_CV_data(target_fold,tr_x,tr_y):
            
            
            for i in range(7):
                
                categorical_labeled_index=np.where((i*12<=tr_pop[labeled_index])*(tr_pop[labeled_index]<=(i+1)*12-1))[0]
                
                categorical_label_index=np.where((i*12<=tr_pop[label_index])*(tr_pop[label_index]<=(i+1)*12-1))[0]
                
                now_labeled_index=(np.arange(len(tr_x))[labeled_index])[categorical_labeled_index]
                now_label_index=(np.arange(len(tr_x))[label_index])[categorical_label_index]
                
                
                tr_target_encode[now_labeled_index]=((np.eye(11)[tr_y])[now_label_index]).mean(axis=0)
                
                
                
               
        for i in range(7):
            
            labeled_indexs=np.where((i*12<=val_pop)*(val_pop<=(i+1)*12-1))[0]
                
            label_indexs=np.where((i*12<=tr_pop)*(tr_pop<=(i+1)*12-1))[0]
            
            val_target_encode[labeled_indexs]=(np.eye(11)[tr_y])[label_indexs].mean(axis=0)
            
        for i in range(7):
            
            labeled_indexs=np.where((i*12<=test_raw_pop)*(test_raw_pop<=(i+1)*12-1))[0]
                
            label_indexs=np.where((i*12<=train_raw_pop)*(train_raw_pop<=(i+1)*12-1))[0]
            
            test_target_encode[labeled_indexs]=((np.eye(11)[train_y])[label_indexs]).mean(axis=0)
            
            
            
        tr_target_encode*=param
        val_target_encode*=param
        test_target_encode*=param
        
        
            
        tr_x=np.concatenate([tr_x,tr_target_encode],axis=1)
        val_x=np.concatenate([val_x,val_target_encode],axis=1)
        now_test_x=np.concatenate([test_x,test_target_encode],axis=1)
       
        
        
        knc=KNC(n_neighbors=k,weights=lambda x:1/(x)**2.1)
    
        knc.fit(tr_x,tr_y)
        
        pred=knc.predict_proba(val_x)
        
        
        
        out_train_y[val_index]=pred
        
        
        
        score+=f1_score(val_y,pred.argmax(axis=1),average='macro')
        
        
        preds+=knc.predict_proba(now_test_x)
        
    preds/=n_fold
    print(k,score/n_fold)
    
    df_train_y=pd.DataFrame(out_train_y)
    df_preds=pd.DataFrame(preds)
    
    data_ver=6
    df_train_y.to_csv('/Users/nakaharakan/Documents/signate_music/models/KNN/KNN'+str(data_ver)+'/'+str(data_ver)+'KNN_train_x.csv',index=False,header=False)
    df_preds.to_csv('/Users/nakaharakan/Documents/signate_music/models/KNN/KNN'+str(data_ver)+'/'+str(data_ver)+'KNN_test_x.csv',index=False,header=False)

   


        
    preds=preds.argmax(axis=1)
    
    
        
    
    sol=pd.DataFrame({'id':[i+4046 for i in range(4046)],'ans':preds})
    
    sol['ans']=sol['ans'].map(lambda x:int(x))
    
    sol.to_csv('/Users/nakaharakan/Documents/signate_music/models/KNN/KNN'+str(data_ver)+'/'+str(data_ver)+'KNN.csv',index=False,header=False)
    
    
    
    return sol
    
    
def param_decide(param=float()):
    
    for i in range(10):
        
        fitter(i+1,8,param)
        




    
        






































