# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:22:48 2020

@author: nicol
"""
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DropoutNet(nn.Module):
        
    def __init__(self, hidden_size , dim_input , dim_output , p):
        super().__init__()
        self.fc1 = nn.Linear(dim_input , hidden_size)
        self.fc2 = nn.Linear(hidden_size , dim_output)
        self.p = p
                             
        
    def forward(self, x):
        return self.fc2(F.dropout(F.relu(self.fc1(x)) , self.p))
    
class DropoutReg():
    
    def __init__(self, X_train , y_train , X_test , net , batch_size):
        self.net = net
        self.batch_size = batch_size
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.pred, self.pred_mean , self.pred_std = None , None , None
        
    def create_batches(self):
        torch_train_dataset = data.TensorDataset(self.X_train, self.y_train) 
        self.batches = data.DataLoader(torch_train_dataset, batch_size=self.batch_size)
        
    def train(self , epochs , optimizer , criterion):       
        self.net.train()        
        self.create_batches()
        for epoch in range(int(epochs)):
            for local_batch, local_labels in self.batches:
                optimizer.zero_grad()
                output = self.net(local_batch).squeeze()
                loss = criterion(output, local_labels)
                loss.backward()
                optimizer.step()
        return
    
    def predict(self , samples):
        
        self.net.eval()
        self.net.training = True  
        self.pred = torch.zeros((self.X_test.shape[0] , self.y_train.unsqueeze(dim=1).shape[1] , samples))
        for s in range(samples):
            self.pred[: , : , s] = self.net(self.X_test).detach()
            
        self.pred_mean = torch.mean(self.pred , dim = 2).squeeze()
        self.pred_std = torch.std(self.pred , dim = 2).squeeze()
        
        return self.pred_mean , self.pred_std
    
    def plot_results(self):
        
        X_test = self.X_test.squeeze().numpy()
        y_pred = self.pred_mean.squeeze().numpy()
        std_pred = self.pred_std.squeeze().numpy()
        
        plt.fill_between(X_test, y_pred-std_pred, y_pred+std_pred, color='indianred', label='1 std. int.')
        plt.fill_between(X_test, y_pred-std_pred*2, y_pred-std_pred, color='lightcoral')
        plt.fill_between(X_test, y_pred+std_pred*1, y_pred+std_pred*2, color='lightcoral', label='2 std. int.')
        plt.fill_between(X_test, y_pred-std_pred*3, y_pred-std_pred*2, color='mistyrose')
        plt.fill_between(X_test, y_pred+std_pred*2, y_pred+std_pred*3, color='mistyrose', label='3 std. int.')
        
        plt.scatter(self.X_train.numpy() , self.y_train.numpy() , color = 'red' , marker = 'x' , label = "trainig points")
        plt.plot(X_test , y_pred , color = 'blue', label = "prediction")
        return
    
            