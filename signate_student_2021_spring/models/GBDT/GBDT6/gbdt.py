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
import xgboost as xgb
from sklearn.metrics import log_loss



data_ver=6

train_path='/Users/nakaharakan/Documents/signate_music/process_data/process_data'+str(data_ver)+'/processed_train'+str(data_ver)+'.csv'
test_path='/Users/nakaharakan/Documents/signate_music/process_data/process_data'+str(data_ver)+'/processed_test'+str(data_ver)+'.csv'


train_raw_pop=np.array(pd.read_csv('/Users/nakaharakan/Documents/signate_music/train.csv')['popularity'])
test_raw_pop=np.array(pd.read_csv('/Users/nakaharakan/Documents/signate_music/test.csv')['popularity'])


def F1(preds,target):
    
    preds=np.array(preds)
    target=np.array(target)
    
    target
    
    
    
    
    
    print(preds)
    
    pred_label=preds.reshape(len(np.unique(target)),-1).argmax(axis=0)
    
    f1=f1_score(target,pred_label,average='macro')
    
    return 'f1',f1


def stratified_CV_data(n_fold,train_x,train_y):
    
    skf=SKF(n_splits=n_fold,random_state=None,shuffle=False)
    
    return skf.split(train_x,train_y)


def CV2csv(n_fold,target_fold):
    
    preds=np.zeros((4046,11))
    
    params={"objective": "multi:softprob", 
        "eval_metric": "mlogloss",
        "eta": 0.05, 
        "max_depth": 6, 
        "min_child_weight": 1, 
        "subsample": 1, 
        "colsample_bytree": 1,
        "num_class": 11
        }
    
    
    train_x=pd.read_csv(train_path)
    test_x=pd.read_csv(test_path)
    
    train_y=train_x['genre']
    
    #print(train_x.columns)
    
    del train_x['genre'],train_x['Unnamed: 0'],test_x['Unnamed: 0']
    
    print('eta:'+str(params['eta']),'max_depth:'+str(params['max_depth']))
    
    
    score=0
    
    train_x=np.array(train_x)
    test_x=np.array(test_x)
    train_y=np.array(train_y)
    
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
            
            
            
            
        tr_x=np.concatenate([tr_x,tr_target_encode],axis=1)
        val_x=np.concatenate([val_x,val_target_encode],axis=1)
        now_test_x=np.concatenate([test_x,test_target_encode],axis=1)
       
        
        print('=========='*5)
        
        dtrain=xgb.DMatrix(tr_x,label=tr_y)
        dvalid=xgb.DMatrix(val_x,label=val_y)
        dtest=xgb.DMatrix(now_test_x)
        
        watchlist=[(dtrain,'train'),(dvalid,'eval')]
        
        model=xgb.train(params,dtrain,10000,early_stopping_rounds=200,evals=watchlist,verbose_eval=1000)
        
        score+=f1_score(val_y,
                       model.predict(dvalid,
                                     ntree_limit=model.best_ntree_limit).argmax(axis=1),average='macro')
        
        
        
        preds=model.predict(dtest,ntree_limit=model.best_ntree_limit)
        
    print('=========='*5)
        
    print('score:'+str(score/n_fold))
        
    xgb.plot_importance(model)
        
    
    
    
        
    preds=preds.argmax(axis=1)
    
    
    
    sol=pd.DataFrame({'id':[i+4046 for i in range(4046)],'ans':preds})
    
    sol['ans']=sol['ans'].map(lambda x:int(x))
    
    sol.to_csv('/Users/nakaharakan/Documents/signate_music/models/GBDT/GBDT'+str(data_ver)+'/'+str(data_ver)+'gbdt.csv',index=False,header=False)
    
    return sol
        
    
        






































