#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:07:36 2021

@author: nakaharakan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import f1_score

data_ver=1

train_path='/Users/nakaharakan/Documents/signate_music/process_data/process_data'+str(data_ver)+'/processed_train'+str(data_ver)+'.csv'
test_path='/Users/nakaharakan/Documents/signate_music/process_data/process_data'+str(data_ver)+'/processed_test'+str(data_ver)+'.csv'

import torch
from torch import tensor,LongTensor
from torch.utils.data import TensorDataset,DataLoader
from torch.nn import functional as F,Module,Parameter,Linear,Dropout
from torch.optim import Adam

class simpleNN(Module):
    
    def __init__(self,n_in):
        
        super(simpleNN,self).__init__()
        self.fc1=Linear(n_in,32)
        self.dr1=Dropout(0.3)
        self.fc2=Linear(32,16)
        self.dr2=Dropout(0.)
        self.fc3=Linear(16,11)
        
    def forward(self,x):
        
        x=F.relu(self.fc1(x))
        
        #x=self.dr1(x)
        
        x=F.relu(self.fc2(x))
        
        #x=self.dr2(x)
        
        x=F.softmax(self.fc3(x),dim=1)
        
        return x
        
        
        
        
        



def stratified_CV_data(n_fold,train_x,train_y):
    
    skf=SKF(n_splits=n_fold,random_state=None,shuffle=False)
    
    return skf.split(train_x,train_y)


def CV2csv(n_fold):
    
    preds=np.zeros((4046,11))
    
    
    train_x=pd.read_csv(train_path)
    test_x=pd.read_csv(test_path)
    train_y=train_x['genre']
    
    del train_x['genre'],train_x['Unnamed: 0'],test_x['Unnamed: 0']
    
    train_x=np.array(train_x)
    test_x=np.array(test_x)
    train_y=np.array(train_y)
    
    test_x=tensor(test_x)
    
    max_loss=[]
    
    output_x=np.zeros((4046,11))
    
    for train_index,val_index in stratified_CV_data(n_fold,train_x,train_y):
        
        model=simpleNN(train_x.shape[1])
        
        optim=Adam(model.parameters(),lr=0.005)
        
        
        
        tr_x=tensor(train_x[train_index],dtype=float)
        tr_y=tensor(np.eye(11)[train_y[train_index]])
        fit_set=TensorDataset(tr_x,tr_y)
        fit_loader=DataLoader(fit_set,batch_size=32,shuffle=True)
        
        val_x=tensor(train_x[val_index],dtype=float)
        
        
        
        loss_hist=[]
        
        for epoch in range(17):
            
            model.train()
        
            for data,targets in fit_loader:
                
                
            
                optim.zero_grad()
                
                y=model(data.float())
            
                loss=-(targets*((y+1e-8).log())).sum(axis=1).mean()
                
                loss.backward()
                
                optim.step()
                
            model.eval()
            
            f_score=f1_score(train_y[val_index],model(val_x.float()).argmax(dim=1).detach().numpy(),
            average='macro')
            
            
                
            
            print(f_score)
                
            loss_hist.append(f_score)
            
        print('max:'+str(max(loss_hist)))
        
        max_loss.append(max(loss_hist))
        
        output_x[val_index]=model(val_x.float()).detach().numpy()
        
        preds+=model(test_x.float()).detach().numpy()
        
    df_train_y=pd.DataFrame(output_x)
    df_preds=pd.DataFrame(preds/n_fold)
    
    df_train_y.to_csv('/Users/nakaharakan/Documents/signate_music/models/simpleNN/simpleNN'+str(data_ver)+'/'+str(data_ver)+'simpleNN_train_x.csv',index=False)
    df_preds.to_csv('/Users/nakaharakan/Documents/signate_music/models/simpleNN/simpleNN'+str(data_ver)+'/'+str(data_ver)+'simpleNN_test_x.csv',index=False)

        
    print('CV_loss:'+str(sum(max_loss)/len(max_loss)))
    
    return preds
    
    
def ave_sol(n=int()):
    
    preds=np.zeros((4046,11))
    
    for i in range(n):
        
        preds+=CV2csv(10)
        
    preds=preds.argmax(axis=1)
    
    
    
    sol=pd.DataFrame({'id':[i+4046 for i in range(4046)],'ans':preds})
    
    sol['ans']=sol['ans'].map(lambda x:int(x))
    
    sol.to_csv('/Users/nakaharakan/Documents/signate_music/models/simpleNN/simpleNN'+str(data_ver)+'/'+str(data_ver)+'simpleNN.csv',index=False,header=False)
    
    return sol


                
        
                
                
            
            
            
            
            
            
            
            
            