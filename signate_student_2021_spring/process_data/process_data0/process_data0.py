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


train_path='/Users/nakaharakan/Documents/signate_music/train.csv'
test_path='/Users/nakaharakan/Documents/signate_music/test.csv'


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



    

    all_data=pd.get_dummies(all_data)
    
    print(all_data.isnull().sum())
    print(all_data.info())
    
    train=all_data[:4046]
    test=all_data[4046:]

    train['genre']=pd.read_csv(train_path)['genre']

    train.to_csv('processed_train0.csv')
    test.to_csv('processed_test0.csv')
    
    


    
    
    
    
    


def show_data():
    
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    
    del train['genre']
    del train['index']
    rcParams['figure.figsize']=10,10
    
    train.hist()
    
    plt.tight_layout()
    
    plt.show()
    
    
    
    





























