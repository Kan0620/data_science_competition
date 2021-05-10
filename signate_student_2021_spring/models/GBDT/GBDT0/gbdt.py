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



data_ver=0

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


def CV2csv(n_fold):
    
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
    output_train=np.zeros((4046,11))
    
    
    
    print('eta:'+str(params['eta']),'max_depth:'+str(params['max_depth']))
    print(del_columns)
    
    score=0
    
    for train_index,val_index in stratified_CV_data(n_fold,train_x,train_y):
        print('=========='*5)
        
        dtrain=xgb.DMatrix(train_x.iloc[train_index],label=train_y.iloc[train_index])
        dvalid=xgb.DMatrix(train_x.iloc[val_index],label=train_y.iloc[val_index])
        dtest=xgb.DMatrix(test_x)
        
        watchlist=[(dtrain,'train'),(dvalid,'eval')]
        
        model=xgb.train(params,dtrain,10000,early_stopping_rounds=200,evals=watchlist,verbose_eval=1000)
        
        score+=f1_score(train_y.iloc[val_index],
                       model.predict(xgb.DMatrix(train_x.iloc[val_index]),
                                     ntree_limit=model.best_ntree_limit).argmax(axis=1),average='macro')
        
        output_train[val_index]=model.predict(xgb.DMatrix(train_x.iloc[val_index]),
                                     ntree_limit=model.best_ntree_limit)
        
        preds+=model.predict(dtest,ntree_limit=model.best_ntree_limit)
        
    print('=========='*5)
        
    print('score:'+str(score/n_fold))
        
        #xgb.plot_importance(model)
        
    
    df_train_y=pd.DataFrame(output_train)
    df_preds=pd.DataFrame(preds/n_fold)
    
    df_train_y.to_csv('/Users/nakaharakan/Documents/signate_music/models/GBDT/GBDT'+str(data_ver)+'/'+str(data_ver)+'GBDT_train_x.csv',index=False)
    df_preds.to_csv('/Users/nakaharakan/Documents/signate_music/models/GBDT/GBDT'+str(data_ver)+'/'+str(data_ver)+'GBDT_test_x.csv',index=False)

    
        
    preds=preds.argmax(axis=1)
    
    
    
    sol=pd.DataFrame({'id':[i+4046 for i in range(4046)],'ans':preds})
    
    sol['ans']=sol['ans'].map(lambda x:int(x))
    
    sol.to_csv('/Users/nakaharakan/Documents/signate_music/models/GBDT/GBDT'+str(data_ver)+'/'+str(data_ver)+'gbdt.csv',index=False,header=False)
    
    return sol





def CV4ens(n_fold=int()):
    
    preds=np.zeros((4046,11))
    
    params={"objective": "multi:softprob", 
        "eval_metric": "mlogloss",
        "eta": 0.1, 
        "max_depth": 6, 
        "min_child_weight": 1, 
        "subsample": 1, 
        "colsample_bytree": 1,
        "num_class": 11
        }
    
    
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
    
    
    print('eta:'+str(params['eta']),'max_depth:'+str(params['max_depth']))
    print(del_columns)
    
    score=0
    
    for train_index,val_index in stratified_CV_data(n_fold,train_x,train_y):
        print('=========='*5)
        
        dtrain=xgb.DMatrix(train_x.iloc[train_index],label=train_y.iloc[train_index])
        dvalid=xgb.DMatrix(train_x.iloc[val_index],label=train_y.iloc[val_index])
        dtest=xgb.DMatrix(test_x)
        
        watchlist=[(dtrain,'train'),(dvalid,'eval')]
        
        model=xgb.train(params,dtrain,10000,early_stopping_rounds=200,evals=watchlist,verbose_eval=1000)
        
        score+=f1_score(train_y.iloc[val_index],
                       model.predict(xgb.DMatrix(train_x.iloc[val_index]),
                                     ntree_limit=model.best_ntree_limit).argmax(axis=1),average='macro')
        
        preds=model.predict(dtest,ntree_limit=model.best_ntree_limit)
        
        xgb.plot_importance(model)
        
        print(model.get_fscore())
        
        
        
    print('=========='*5)
        
    print('score:'+str(score/n_fold))
        
    plt.show()
        
    sol=pd.DataFrame(preds)
    
    sol.to_csv('/Users/nakaharakan/Documents/signate_music/models/GBDT/GBDT'+str(data_ver)+'/'+str(data_ver)+'gbdt_ens_lr1.csv',index=False,header=False)
    
    return model
    
    
    
    
    
    
        
    
        






































