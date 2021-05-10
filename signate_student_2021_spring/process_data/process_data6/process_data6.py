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
from scipy.stats import rankdata


train_path='/Users/nakaharakan/Documents/signate_music/train.csv'
test_path='/Users/nakaharakan/Documents/signate_music/test.csv'

def tempo_num(x=str()):
        
    return (float(x.split('-')[0])+float(x.split('-')[1]))/2


def create_data():
    
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    
    del train['genre']
    
    all_data=pd.concat((train,test))

    del all_data['index']
    
    print(all_data.dtypes)
    
    print(all_data.isnull().sum())
    
    print(all_data[['popularity','duration_ms','acousticness','positiveness','danceability','loudness',\
'energy','liveness','speechiness','instrumentalness']].dropna().apply(lambda x:skew(x)))
    
    all_data=all_data.fillna(all_data.mean())
    '''
    
    all_data['binary_inst']=all_data['instrumentalness'].apply(lambda x: int(x>0.5))
    
    all_data['binary_live']=all_data['liveness'].apply(lambda x: int(x>0.8))
    
    all_data['categorical_speech_0']=all_data['speechiness'].apply(lambda x: int(x<0.33))
    
    all_data['categorical_speech_1']=all_data['speechiness'].apply(lambda x: int(0.33<x and x<0.66))
    
    all_data['categorical_speech_2']=all_data['speechiness'].apply(lambda x: int(0.66<x))
    
    all_data['binary_positive']=all_data['positiveness'].apply(lambda x: int(x>0.5))
    
    '''
    
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
    
    all_data[['duration_ms','acousticness','liveness','speechiness','instrumentalness']]=\
    np.log1p(all_data[['duration_ms','acousticness','liveness','speechiness','instrumentalness']])
    
    
    all_data[['loudness']]=np.log1p(-all_data[['loudness']])
    

    print(all_data[['duration_ms','acousticness','liveness','speechiness','instrumentalness','loudness']]\
          .dropna().apply(lambda x:skew(x)))
    
    all_data['tempo']=all_data['tempo'].apply(tempo_num)
    
    
    all_data['fast']=all_data['tempo']/all_data['duration_ms']\
    *np.exp(-all_data['loudness']/10)
    
    all_data['slow']=all_data['duration_ms']/all_data['tempo']/(all_data['positiveness']+1)
    
    del all_data['tempo']
    
    print(all_data.isna().sum())

    all_data=pd.get_dummies(all_data)
    
    norm_list=['popularity','duration_ms','acousticness','positiveness','loudness',
               'danceability','energy','liveness','speechiness','instrumentalness','fast']
    
    all_data[norm_list]=(all_data[norm_list]-all_data[norm_list].mean())/all_data[norm_list].std()
    
    
    print(all_data.info())
    
    train=all_data[:4046]
    test=all_data[4046:]

    train['genre']=pd.read_csv(train_path)['genre']

    train.to_csv('processed_train6.csv')
    test.to_csv('processed_test6.csv')
    
    


    
    
    
    
    


def show_data():
    
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    
    del train['genre']
    del train['index']
    rcParams['figure.figsize']=10,10
    
    train.hist()
    
    plt.tight_layout()
    
    plt.show()
    
    
    
    





























