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



data_ver=9

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


def CV2csv(n_fold,target_fold,key_cols=list()):
    
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
    num_leaves=35
    
    print('colsample_bytree:{} num_leaves:{}'.format(colsample_bytree,num_leaves))
    
    score=0
    
    for train_index,val_index in stratified_CV_data(n_fold,train_x,train_y):
        
        
        
        
        print('=========='*5)
        
        tr_x=train_x.iloc[train_index]
        tr_y=train_y.iloc[train_index]
        
        val_x=train_x.iloc[val_index]
        val_y=train_y.iloc[val_index]
        
        one_hot_tr_y=pd.DataFrame(np.eye(11)[np.array(tr_y)])
        one_hot_all_y=pd.DataFrame(np.eye(11)[np.array(train_y)])
        
        
        one_hot_tr_y=one_hot_tr_y.set_index(train_index)
        print(one_hot_tr_y.shape,train_index.shape)
        
        for key in key_cols:
                
                
                for i in range(11):
                    
                    tr_x[key+'_target'+str(i)]=np.zeros(len(tr_x))
                    val_x[key+'_target'+str(i)]=np.zeros(len(val_x))
                    test_x[key+'_target'+str(i)]=np.zeros(len(test_x))
        
        #target encoding
        for label_index,labeled_index in stratified_CV_data(target_fold,tr_x,tr_y):
            
            for key in key_cols:
                
                target_column=[key+'_target'+str(i) for i in range(11)]
                
                
                
                for i in (tr_x.iloc[labeled_index])[key].unique():
                    
                    
                    
                    labeled_cond_1=[i in labeled_index for i in range(len(tr_x))]
                    labeled_cond_2=tr_x[key]==i
                    labeled_cond=labeled_cond_1*labeled_cond_2
                    
                    
                    label_cond_1=[i in label_index for i in range(len(tr_x))]
                    label_cond_2=tr_x[key]==i
                    label_cond=label_cond_1*label_cond_2
                    
                    for j,target_col in enumerate(target_column):
                        
                        tr_x.loc[labeled_cond,target_col]=one_hot_tr_y.loc[label_cond,j].mean()
                        
                
        for key in key_cols:
            
            target_column=[key+'_target'+str(i) for i in range(11)]
            
            for i in val_x[key].unique():
                
                for j,target_col in enumerate(target_column):
                
                    val_x.loc[val_x[key]==i,target_col]=one_hot_tr_y.loc[tr_x[key]==i,j].mean()
        
        for key in key_cols:
            
            target_column=[key+'_target'+str(i) for i in range(11)]
            
            for i in test_x[key].unique():
                
                for j,target_col in enumerate(target_column):
                
                    test_x.loc[test_x[key]==i,target_col]=one_hot_all_y.loc[train_x[key]==i,j].mean()
        
        
        
        for key in key_cols:
            
            for i in range(11):
                
                tr_x[key+'_delta_target'+str(i)]=tr_x[str(i)+'_prob']-tr_x[key+'_target'+str(i)]
                val_x[key+'_delta_target'+str(i)]=val_x[str(i)+'_prob']-val_x[key+'_target'+str(i)]
                test_x[key+'_delta_target'+str(i)]=test_x[str(i)+'_prob']-test_x[key+'_target'+str(i)]
                
                tr_x[key+'_sum_target'+str(i)]=tr_x[str(i)+'_prob']+tr_x[key+'_target'+str(i)]
                val_x[key+'_sum_target'+str(i)]=val_x[str(i)+'_prob']+val_x[key+'_target'+str(i)]
                test_x[key+'_sum_target'+str(i)]=test_x[str(i)+'_prob']+test_x[key+'_target'+str(i)]
                
        for key in key_cols:
            
            for i in range(11):
                
                del tr_x[key+'_target'+str(i)],val_x[key+'_target'+str(i)],test_x[key+'_target'+str(i)]
        
        
        
        print('data loaded')
        
        dtrain=lgb.Dataset(tr_x,tr_y)
        dvalid=lgb.Dataset(val_x,label=val_y,reference=dtrain)
        
        
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
        
        model.fit(X=tr_x,
                  y=tr_y,
                  eval_set=[(val_x,val_y)],
                  early_stopping_rounds=100,
                  verbose=False
                  )
        
        now_score=f1_score(val_y,
                       model.predict(val_x,num_iteration=model.best_iteration_),
                       average='macro')
        
        print(now_score)
        
        score+=now_score
        
        
        
        output_train[val_index]=model.predict_proba(val_x,num_iteration=model.best_iteration_)
        
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

























