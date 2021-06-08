#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:24:18 2020

@author: nakaharakan
"""


import pandas as pd
from scipy.stats import skew
import numpy as np

def goal1(x=str()):
    
    price=x.split('-')[0]
    
    
    
    
    if price=='100000+':
        return 0
    else:
        
        return (int((int(price)-1)/1000))

def goal2(x=str()):
    
    price=x.split('-')[0]
    if price=='100000+':
        return 1
    else:
        return 0
    
def n_html(x=str()):
    
    
    return len(x)

def hatena(x=str()):
    
    return len(x.split('?'))-1

def bikkuri(x=str()):
    
    return len(x.split('!'))-1

def html2data(x=str()):
    
    fig=len(x.split('<figure>'))
    ht=len(x.split('http://'))
    n=0
    for one_str in x.split('<p>'):
        
        if '</p>' in one_str:
            
            n+=1
                
    return pd.Series([len(x.split('<p>'))-1,fig-1,len(x.split('<div>'))-1,ht-1,\
                        len(x.split('</a>'))-1,n])

def GDP_pop(row):
    
    return float(row['GDP'])/float(row['pop'])

def coun_data(x=str()):
    
    if x=='AT':        
        return pd.Series([4553,8955,'2'])        
    elif x=='AU':        
        return pd.Series([14340,25203,'2'])        
    elif x=='BE':        
        return pd.Series([5428,11539,'2'])    
    elif x=='CA':
        return pd.Series([17130,37411,'0'])
    elif x=='CH':
        return pd.Series([7051,8591,'2'])
    elif x=='DE':
        return pd.Series([39480,83517,'0'])
    elif x=='DK':
        return pd.Series([3557,5806,'2'])
    elif x=='ES':
        return pd.Series([14190,46737,'2'])
    elif x=='FR':
        return pd.Series([27780,65130,'0'])
    elif x=='GB':
        return pd.Series([28550,67530,'0'])
    elif x=='HK':
        return pd.Series([3627,7436,'1'])
    elif x=='IE':
        return pd.Series([3825,4882,'2'])
    elif x=='IT':
        return pd.Series([20840,60550,'0'])
    elif x=='JP':
        return pd.Series([49710,126860,'0'])
    elif x=='LU':
        return pd.Series([708,613,'2'])
    elif x=='MX':
        return pd.Series([12210,127576,'2'])
    elif x=='NL':
        return pd.Series([9137,17097,'2'])
    elif x=='NO':
        return pd.Series([4342,5379,'2'])
    elif x=='NZ':
        return pd.Series([2049,4783,'2'])
    elif x=='SE':
        return pd.Series([5561,10036,'2'])
    elif x=='SG':
        return pd.Series([3642,5804,'1'])
    elif x=='US':
        return pd.Series([200000,329065,'0'])
    else:
        print('error')

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

del train['state']

print(test.info())

all_data=pd.concat((train,test))

del all_data['id']
#del all_data['category2']

all_data['goal1']=all_data['goal'].map(goal1)
all_data['goal2']=all_data['goal'].map(goal2)
del all_data['goal']

all_data[['p','fig','div','ht','a','n']]=all_data['html_content'].apply(html2data)

all_data['!']=all_data['html_content'].map(bikkuri)
all_data['?']=all_data['html_content'].map(hatena)

print(all_data['n'].max())

all_data['html_content']=all_data['html_content'].map(n_html)


print(all_data[['goal1','html_content','duration','p','fig','div','ht','a','n','!','?']].apply(lambda x:skew(x)))

all_data[['goal1','html_content','p','fig','div','ht','a','n','!','?']]=\
np.log1p(all_data[['goal1','html_content','p','fig','div','ht','a','n','!','?']])

print(all_data[['goal1','html_content','duration','p','fig','div','ht','a','n','!','?']].apply(lambda x:skew(x)))

all_data['goal1']=all_data['goal1'].map(lambda x:int(10*x))

print(all_data[['goal1','html_content']].apply(lambda x:skew(x)))

all_data['goal1']=all_data['goal1'].map(lambda x:str(x))

now=all_data['duration']


print(all_data.info())






all_data['html_content']=(all_data['html_content']-all_data['html_content'].min())/\
(all_data['html_content'].max()-all_data['html_content'].min())

all_data['p']=(all_data['p']-all_data['p'].min())/\
(all_data['p'].max()-all_data['p'].min())

all_data['fig']=(all_data['fig']-all_data['fig'].min())/\
(all_data['fig'].max()-all_data['fig'].min())

all_data['div']=(all_data['div']-all_data['div'].min())/\
(all_data['div'].max()-all_data['div'].min())

all_data['ht']=(all_data['ht']-all_data['ht'].min())/\
(all_data['ht'].max()-all_data['ht'].min())

all_data['a']=(all_data['a']-all_data['a'].min())/\
(all_data['a'].max()-all_data['a'].min())

all_data['n']=(all_data['n']-all_data['n'].min())/\
(all_data['n'].max()-all_data['n'].min())

all_data['!']=(all_data['!']-all_data['!'].min())/\
(all_data['!'].max()-all_data['!'].min())

all_data['duration']=(all_data['duration']-all_data['duration'].min())/\
(all_data['duration'].max()-all_data['duration'].min())




all_data=pd.get_dummies(all_data)

train=all_data[:10545]
test=all_data[10545:]

train['state']=pd.read_csv('train.csv')['state']

train.to_csv('processed_train10.csv')
test.to_csv('processed_test10.csv')

print(train.info())






