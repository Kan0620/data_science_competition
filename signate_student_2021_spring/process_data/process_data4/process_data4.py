#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:11:31 2021

@author: nakaharakan
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pylab import rcParams


train_path='/Users/nakaharakan/Documents/signate_music/train.csv'
test_path='/Users/nakaharakan/Documents/signate_music/test.csv'

def tempo_num(x=str()):
        
    return (float(x.split('-')[0])+float(x.split('-')[1]))/2


def create_data(n_components=int()):
    
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    
    del train['genre']
    
    all_data=pd.concat((train,test))

    del all_data['index']
    
    
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
    

    
    all_data['tempo']=all_data['tempo'].apply(tempo_num)
    
    nan_list=['acousticness','positiveness','danceability','energy',
              'liveness','speechiness','instrumentalness']
    
    nan_name=['nan_acousticness','nan_positiveness','nan_danceability',
              'nan_energy','nan_liveness','nan_speechiness','nan_instrumentalness']
    
    all_data[nan_name]=all_data[nan_list].isna()
    
    for index in nan_name:
        
        all_data[index]=all_data[index].apply(lambda x:int(x))
    
    
    
    
    all_data=all_data.fillna(all_data.mean())

    all_data=pd.get_dummies(all_data)
    
    norm_list=['popularity','duration_ms','acousticness','positiveness','danceability','loudness',
               'energy','liveness','speechiness','instrumentalness','tempo']
    
    all_data[norm_list]=(all_data[norm_list]-all_data[norm_list].mean())/all_data[norm_list].std()
    
    pca=PCA(n_components=n_components)
    
    pca.fit(all_data)
    
    all_data=pca.transform(all_data)
    
    #all_data=(all_data-all_data.mean())/all_data.std()
    
    train=pd.DataFrame(all_data[:4046,:])
    test=pd.DataFrame(all_data[4046:,:])

    train['genre']=pd.read_csv(train_path)['genre']

    train.to_csv('processed_train4.csv')
    test.to_csv('processed_test4.csv')
    
    


    
    
    
    
    


def show_data():
    
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    
    del train['genre']
    del train['index']
    rcParams['figure.figsize']=10,10
    
    train.hist()
    
    plt.tight_layout()
    
    plt.show()
    
    
    
    





























