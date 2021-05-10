#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:11:31 2021

@author: nakaharakan
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.cluster import KMeans


train_path='/Users/nakaharakan/Documents/signate_music/train.csv'
test_path='/Users/nakaharakan/Documents/signate_music/test.csv'

def stratified_CV_data(n_fold,train_x,train_y):
    
    skf=SKF(n_splits=n_fold,random_state=None,shuffle=False)
    
    return skf.split(train_x,train_y)

def various_f(index,raw_df,key,about,n_unique):
    
    if index==0:
        
        raw_df['agg_'+about+'_by_'+key+'_mean']=np.zeros(len(raw_df))
        
        for group in range(n_unique):
            
            x=raw_df[raw_df[key]==group][about]
            
            raw_df['agg_'+about+'_by_'+key+'_mean'][raw_df[key]==group]=x.mean()
        
        return raw_df
    
    elif index==1:
        
        raw_df['agg_'+about+'_by_'+key+'_std']=np.zeros(len(raw_df))
        
        for group in range(n_unique):
            
            x=raw_df[raw_df[key]==group][about]
            
            raw_df['agg_'+about+'_by_'+key+'_std'][raw_df[key]==group]=x.std()
        
        return raw_df
    
    elif index==2:
        
        raw_df['agg_'+about+'_by_'+key+'_z']=np.zeros(len(raw_df))
        
        for group in range(n_unique):
            
            x=raw_df[raw_df[key]==group][about]
            
            raw_df['agg_'+about+'_by_'+key+'_z'][raw_df[key]==group]=(x-x.mean())/(x.std()+1e-8)
        
        return raw_df
        
    
    elif index==3:
        
        raw_df['agg_'+about+'_by_'+key+'_max']=np.zeros(len(raw_df))
        
        for group in range(n_unique):
            
            x=raw_df[raw_df[key]==group][about]
            
            raw_df['agg_'+about+'_by_'+key+'_max'][raw_df[key]==group]=x.max()
        
        return raw_df
    
        
    
    elif index==4:
        
        raw_df['agg_'+about+'_by_'+key+'_min']=np.zeros(len(raw_df))
        
        for group in range(n_unique):
            
            x=raw_df[raw_df[key]==group][about]
            
            raw_df['agg_'+about+'_by_'+key+'_min'][raw_df[key]==group]=x.min()
        
        return raw_df
    
    elif index==5:
        
        raw_df['agg_'+about+'_by_'+key+'_minmax-norm']=np.zeros(len(raw_df))
        
        for group in range(n_unique):
            
            x=raw_df[raw_df[key]==group][about]
            
            raw_df['agg_'+about+'_by_'+key+'_minmax-norm'][raw_df[key]==group]=(x-x.min())/(x.max()-x.min())
        
        return raw_df
        
    
    elif index==6:
        
        raw_df['agg_'+about+'_by_'+key+'_max-min']=np.zeros(len(raw_df))
        
        for group in range(n_unique):
            
            x=raw_df[raw_df[key]==group][about]
            
            raw_df['agg_'+about+'_by_'+key+'_max-min'][raw_df[key]==group]=x.max()-x.min()
        
        return raw_df
        
    
    elif index==7:
        
        raw_df['agg_'+about+'_by_'+key+'_q75-q25']=np.zeros(len(raw_df))
        
        for group in range(n_unique):
            
            x=raw_df[raw_df[key]==group][about]
            
            raw_df['agg_'+about+'_by_'+key+'_q75-q25'][raw_df[key]==group]=x.quantile(0.75)-x.quantile(0.25)
        
        return raw_df
        
        
    else:
        
        print('error')
    

def agg_trans(raw_df,key,about,method_list):
    
    n_unique=len(raw_df[key].unique())
    print(key,n_unique)
    
    #mean
    
    for index in method_list:
        
        various_f(index,raw_df,key,about,n_unique)
        
    return raw_df

def categorical_speech(x=float()):
    
    if x<0.33:
        
        return str(0)
    
    elif 0.33<x and x<0.66:
        
        return str(1)
    
    else:
        
        return str(2)
    
def create_data4KM(n_clusters=int()):
    
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    
    del train['genre']
    
    all_data=pd.concat((train,test))

    del all_data['index']
    
    all_data=all_data.fillna(all_data.mean())
    
    all_data[['duration_ms','acousticness','liveness','speechiness','instrumentalness']]=\
    np.log1p(all_data[['duration_ms','acousticness','liveness','speechiness','instrumentalness']])
    
    
    all_data[['loudness']]=np.log1p(-all_data[['loudness']])
    

    all_data['tempo']=all_data['tempo'].apply(tempo_num)
    
    
    all_data['fast']=all_data['tempo']/all_data['duration_ms']\
    *np.exp(-all_data['loudness']/10)
    
    all_data['slow']=all_data['duration_ms']/all_data['tempo']/(all_data['positiveness']+1)
    
    del all_data['tempo']
    
    all_data=pd.get_dummies(all_data)
    
    norm_list=['popularity','duration_ms','acousticness','positiveness','loudness',
               'danceability','energy','liveness','speechiness','instrumentalness','fast']
    
    all_data[norm_list]=(all_data[norm_list]-all_data[norm_list].mean())/all_data[norm_list].std()
    
    np_all_data=np.array(all_data)
    
    KM=KMeans(n_clusters=n_clusters)
    
    KM.fit(np_all_data)
    
    km_d=KM.transform(np_all_data)
    
    km_c=KM.predict(np_all_data).reshape((-1,1))
    
    name=['from{}clus'.format(i) for i in range(n_clusters)]+['clus']
    
    
    
    km=pd.DataFrame(np.concatenate([km_d,km_c],axis=1),columns=name)
    
    
    return km
    
    
    

def create_data4KNN():
    
    
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    
    del train['genre']
    
    all_data=pd.concat((train,test))

    del all_data['index']
    
    
    all_data=all_data.fillna(all_data.mean())
    
    all_data[['duration_ms','acousticness','liveness','speechiness','instrumentalness']]=\
    np.log1p(all_data[['duration_ms','acousticness','liveness','speechiness','instrumentalness']])
    
    
    all_data[['loudness']]=np.log1p(-all_data[['loudness']])
    

    all_data['tempo']=all_data['tempo'].apply(tempo_num)
    
    
    all_data['fast']=all_data['tempo']/all_data['duration_ms']\
    *np.exp(-all_data['loudness']/10)
    
    all_data['slow']=all_data['duration_ms']/all_data['tempo']/(all_data['positiveness']+1)
    
    del all_data['tempo']
    
    all_data=pd.get_dummies(all_data)
    
    norm_list=['popularity','duration_ms','acousticness','positiveness','loudness',
               'danceability','energy','liveness','speechiness','instrumentalness','fast']
    
    all_data[norm_list]=(all_data[norm_list]-all_data[norm_list].mean())/all_data[norm_list].std()
    
    
    train_x=all_data[:4046]
    test_x=all_data[4046:]

    train_y=pd.read_csv(train_path)['genre']
    
    #========================kNN===============================
    
    knn4gbdt_x=np.zeros((4046,22))
    preds=np.zeros((4046,22))
    
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
    
    train_x['fast']=train_x['fast']*0.6
    test_x['fast']=test_x['fast']*0.6
    
    
    train_x['slow']=train_x['slow']*4.2
    test_x['slow']=test_x['slow']*4.2
    
    
    train_x=np.array(train_x)
    test_x=np.array(test_x)
    train_y=np.array(train_y)
    
    score=0
    
    
    for train_index,val_index in stratified_CV_data(8,train_x,train_y):
        
        tr_x=train_x[train_index]
        tr_y=train_y[train_index]
        
        val_x=train_x[val_index]
        val_y=train_y[val_index]
    
        knc=KNC(n_neighbors=6, weights=lambda x:1/(x)**2.1)
    
        knc.fit(tr_x,tr_y)
        
        pred=knc.predict_proba(val_x)
        
        nei_ds,nei_indexs=knc.kneighbors(val_x)
        
        nei_indexs=tr_y[nei_indexs]
        
        sum_d=np.zeros((len(val_x),11))
        
        for i,(nei_d,nei_index) in enumerate(zip(nei_ds,nei_indexs)):
            
            for a_d,a_index in zip(nei_d,nei_index):
                
                sum_d[i,a_index]+=1/(a_d+1e-8)
                
        
        
        knn4gbdt_x[val_index,:11]=pred
        knn4gbdt_x[val_index,11:]=sum_d
        
        
        score+=f1_score(val_y,pred.argmax(axis=1),average='macro')
        
        
        nei_ds,nei_indexs=knc.kneighbors(test_x)
        
        nei_indexs=tr_y[nei_indexs]
        
        sum_d=np.zeros((len(test_x),11))
        
        for i,(nei_d,nei_index) in enumerate(zip(nei_ds,nei_indexs)):
            
            for a_d,a_index in zip(nei_d,nei_index):
                
                sum_d[i,a_index]+=1/(a_d+1e-8)
                
        
        
        preds[np.arange(len(test_x)),:11]+=knc.predict_proba(test_x)
        preds[np.arange(len(test_x)),11:]+=sum_d
        
    
        
    preds/=8
    
   
    print(score/8)
    
    name=['{}_prob'.format(i) for i in range(11)]+['{}_d'.format(i) for i in range(11)]
    
    knn4gbdt_x=pd.DataFrame(knn4gbdt_x,columns=name)
    preds=pd.DataFrame(preds,columns=name)
    
    return knn4gbdt_x,preds

    





def tempo_num(x=str()):
        
    return (float(x.split('-')[0])+float(x.split('-')[1]))/2


def create_data():
    
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    
    del train['genre']
    
    all_data=pd.concat((train,test))

    all_data.reset_index(drop=True, inplace=True)
    
    
    km=create_data4KM(8)
    
    km.reset_index(drop=True, inplace=True)
    
    all_data=pd.concat([all_data,km],axis=1)
    
    tr_knn,test_knn=create_data4KNN()
    
    
    knn_data=pd.concat([tr_knn,test_knn],axis=0)
    
    knn_data.reset_index(drop=True, inplace=True)
    
    all_data=pd.concat([all_data,knn_data],axis=1)
    
    del all_data['index']
    
    all_data=all_data.fillna(all_data.mean())
    
    
    all_data['binary_inst']=all_data['instrumentalness'].apply(lambda x: int(x>0.5))
    
    all_data['binary_live']=all_data['liveness'].apply(lambda x: int(x>0.8))
    
    all_data['categorical_speech']=all_data['speechiness'].apply(categorical_speech)
    
    all_data['binary_positive']=all_data['positiveness'].apply(lambda x: int(x>0.5))
    
    all_data[['duration_ms','acousticness','liveness','speechiness','instrumentalness']]=\
    np.log1p(all_data[['duration_ms','acousticness','liveness','speechiness','instrumentalness']])
    
    all_data['tempo_num']=all_data['tempo'].apply(tempo_num)
    
    all_data[['loudness']]=np.log1p(-all_data[['loudness']])
    
    all_data['fast']=all_data['tempo_num']/all_data['duration_ms']*np.exp(-all_data['loudness']/10)
    
    all_data['slow']=all_data['duration_ms']/all_data['tempo_num']/(all_data['positiveness']+1)
    
    del all_data['tempo_num']
    #del all_data['tempo']
    
    norm_list=['popularity','duration_ms','acousticness','positiveness','danceability','loudness',
               'energy','liveness','speechiness','instrumentalness','fast','slow']
    
    norm_list+=['{}_prob'.format(i) for i in range(11)]+['{}_d'.format(i) for i in range(11)]
    
    
    
    all_data[norm_list]=(all_data[norm_list]-all_data[norm_list].mean())/all_data[norm_list].std()
    
    delta_list=['delta_{}_{}'.format(i,j) for i in range(11) for j in range(i+1,11)]
    
    for i in range(11):
        
        for j in range(i+1,11):
            
            all_data['delta_{}_{}'.format(i,j)]=all_data['{}_prob'.format(i)]-all_data['{}_prob'.format(j)]
            
            
    norm_list+=delta_list
    
    print(norm_list)
    
    
    le_list=['region','tempo']
    
    for le_column in le_list:
    
        le=LabelEncoder()
    
        le=le.fit(all_data[le_column])
    
        all_data[le_column]=le.transform(all_data[le_column])
        
        
    agg_list=['region','tempo']
    
    for agg_key in agg_list:
        
        for column in norm_list:
            
            if agg_key=='tempo':
                
                method_list=[2,3,4,6]
                
            else:
                method_list=[0,1,2,3,4,6]
            agg_trans(all_data,agg_key,column,method_list)
            
    
        
        
    
    all_data=pd.get_dummies(all_data)
    
    
    
        
    
    
    
    
    
    
    '''
    popularity         -0.200278
duration_ms         4.148811 +log
acousticness        0.839868 +log
positiveness        0.177514
danceability       -0.152383
loudness           -1.077024 l+og-
energy             -0.499371
liveness            1.946709 +log
speechiness         2.052792 +log
instrumentalness    2.891089 +log
    '''
    
    
    

    train=all_data[:4046]
    test=all_data[4046:]
    
    print(train.shape)

    train['genre']=pd.read_csv(train_path)['genre']

    train.to_csv('processed_train10.csv')
    test.to_csv('processed_test10.csv')
    
    return all_data.columns


    
    
    
    
    


def show_data():
    
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    
    del train['genre']
    del train['index']
    rcParams['figure.figsize']=10,10
    
    train.hist()
    
    plt.tight_layout()
    
    plt.show()
    
    
    
    





























