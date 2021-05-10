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
import matplotlib.pyplot as plt
import lightgbm as lgb



data_ver=8

train_path='/Users/nakaharakan/Documents/signate_music/process_data/process_data'+str(data_ver)+'/processed_train'+str(data_ver)+'.csv'
test_path='/Users/nakaharakan/Documents/signate_music/process_data/process_data'+str(data_ver)+'/processed_test'+str(data_ver)+'.csv'

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


def CV2csv(n_fold,param):
    
    preds=np.zeros((4046,11))
    
    
    
    
    train_x=pd.read_csv(train_path)
    test_x=pd.read_csv(test_path)
    
    train_y=train_x['genre']
    
    
    
    del train_x['genre'],train_x['Unnamed: 0'],test_x['Unnamed: 0']
    
    del_index=[]
    del_columns=[]
    
    for index in del_index:
        del_columns.append(train_x.columns[index])
    
    
    
    for index in del_columns:
    
        del train_x[index]
        del test_x[index]
    
    
    
    
    '''
    train_x=np.array(train_x)
    train_y=np.array(train_y.map(lambda x:float(x)))
    test_x=np.array(test_x)
    '''
    print(train_x.shape)
    
    output_train=np.zeros((4046,11))
    
    
    
    colsample_bytree=0.5
    num_leaves=param
    
    print('colsample_bytree:{} num_leaves:{}'.format(colsample_bytree,num_leaves))
    
    score=0
    
    for train_index,val_index in stratified_CV_data(n_fold,train_x,train_y):
        print('=========='*5)
        
        dtrain=lgb.Dataset(train_x.iloc[train_index],train_y.iloc[train_index])
        dvalid=lgb.Dataset(train_x.iloc[val_index],label=train_y.iloc[val_index],reference=dtrain)
        
        
        
        
        model=lgb.LGBMClassifier(
                learning_rate=0.01,
                num_leaves=num_leaves,
                n_estimators=10000,
                objective='multiclass',
                class_weight="balanced",
                colsample_bytree=colsample_bytree,
                random_state=2021,
                n_jobs=-1,
                
                )
        
        model.fit(X=train_x.iloc[train_index],
                  y=train_y.iloc[train_index],
                  eval_set=[(train_x.iloc[val_index],train_y.iloc[val_index])],
                  early_stopping_rounds=100,
                  verbose=False
                  )
        
        now_score=f1_score(train_y.iloc[val_index],
                       model.predict(train_x.iloc[val_index],num_iteration=model.best_iteration_),
                       average='macro')
        
        print(now_score)
        
        score+=now_score
        
        
        
        output_train[val_index]=model.predict_proba(train_x.iloc[val_index],
                    num_iteration=model.best_iteration_)
        
        preds+=model.predict_proba(test_x)
    print('=========='*5)
        
    print('score:'+str(score/n_fold))
    
    
    df_train_y=pd.DataFrame(output_train)
    df_preds=pd.DataFrame(preds/n_fold)
    
    df_train_y.to_csv('/Users/nakaharakan/Documents/signate_music/models/GBDT/GBDT'+str(data_ver)+'/'+str(data_ver)+'GBDT_train_x.csv',index=False)
    df_preds.to_csv('/Users/nakaharakan/Documents/signate_music/models/GBDT/GBDT'+str(data_ver)+'/'+str(data_ver)+'GBDT_test_x.csv',index=False)

    
        
    preds=preds.argmax(axis=1)
    
    
    
    sol=pd.DataFrame({'id':[i+4046 for i in range(4046)],'ans':preds})
    
    sol['ans']=sol['ans'].map(lambda x:int(x))
    
    sol.to_csv('/Users/nakaharakan/Documents/signate_music/models/GBDT/GBDT'+str(data_ver)+'/'+str(data_ver)+'gbdt.csv',index=False,header=False)
    
    return sol

























