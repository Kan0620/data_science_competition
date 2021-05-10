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


train_path='/Users/nakaharakan/Documents/signate_music/train.csv'
test_path='/Users/nakaharakan/Documents/signate_music/test.csv'

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

def categorical_pop(x=float()):
    
    for i in range(12):
        
        if i*7<=x and x<=(i+1)*7-1:
            
            return i




def tempo_num(x=str()):
        
    return (float(x.split('-')[0])+float(x.split('-')[1]))/2



def create_data():
    
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    
    del train['genre']
    
    all_data=pd.concat((train,test))

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
    
    all_data['categorical_pop']=all_data['popularity'].apply(categorical_pop)
    
    
    print(all_data['categorical_pop'])
    
    del all_data['tempo_num']
    #del all_data['tempo']
    
    norm_list=['popularity','duration_ms','acousticness','positiveness','danceability','loudness',
               'energy','liveness','speechiness','instrumentalness','fast','slow']
    
    all_data[norm_list]=(all_data[norm_list]-all_data[norm_list].mean())/all_data[norm_list].std()
    
    
    
    le_list=['region']
    
    for le_column in le_list:
    
        le=LabelEncoder()
    
        le=le.fit(all_data[le_column])
    
        all_data[le_column]=le.transform(all_data[le_column])
        
        
    agg_list=['region','categorical_pop']
    
    for agg_key in agg_list:
        
        for column in norm_list:
            
            if agg_key=='region':
                
                method_list=range(8)
                
            elif agg_key=='categorical_pop':
                
                method_list=range(8)
        
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

    train['genre']=pd.read_csv(train_path)['genre']

    train.to_csv('processed_train8.csv')
    test.to_csv('processed_test8.csv')
    
    


    
    
    
    
    


def show_data():
    
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    
    del train['genre']
    del train['index']
    rcParams['figure.figsize']=10,10
    
    train.hist()
    
    plt.tight_layout()
    
    plt.show()
    
    
    
    





























