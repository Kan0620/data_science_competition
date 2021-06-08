#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 17:35:43 2021

@author: nakaharakan
"""

import pandas as pd

df1=pd.DataFrame([['x',1],['y',2],['z',3],['t',3]],columns=['name','age'])


df2=pd.DataFrame([['x',1],['x',5],['y',2],['z',3],['q',4]],columns=['name','high'])


df=pd.merge(df1,df2,left_on='name',right_on='name',how='left')

print(df)


str_1='2020/09/17'
str_2='2020/09/18'
print((pd.to_datetime(str_1)>pd.to_datetime(str_2)))


a=None

if a==None:
    print('u')
    

    
now='2020-07-01 00:50:01 UTC'

month=now[5:7]
day=now[8:10]
hour=now[11:13]
minu=now[14:16]
print(int(month),day,hour,minu,)

print('aa/'.split('/'))
print('aa'.split('/'))
open('/Users/nakaharakan/Documents/readme.txt',mode='w')