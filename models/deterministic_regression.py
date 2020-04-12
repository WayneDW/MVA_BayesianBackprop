# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


class DeterministicNet(nn.Module):
    
    """ Defines a neural network with one hidden layer of size hidden_size. A
        relu activation is applied after the hidden layer.
    """
    
    def __init__(self, hidden_size, dim_input, dim_output):
        super().__init__()
        self.fc1 = nn.Linear(dim_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, dim_output)
        self.layers = [self.fc1, self.fc2]

    def forward(self, x):
        out = F.relu(self.fc1(x))
        return self.fc2(out)

    def weights_dist(self):
        """ Return flatten numpy array containing all the weights of the net """
        return np.hstack(list(map(lambda layer: layer.weight.data.numpy().flatten(), self.layers)))


class DeterministicReg(object):
    
    """ Defines the regression model for a training set (X_train , y_Train),
        a test set X_test and a neural-network net.
    """
    
    def __init__(self, X_train, y_train, X_test, net, batch_size=None):
        self.net = net
        self.batch_size = batch_size
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.pred = None
        self.batches = None

    def create_batches(self):
        torch_train_dataset = data.TensorDataset(self.X_train, self.y_train)
        self.batches = data.DataLoader(torch_train_dataset, batch_size=self.batch_size)

    def train(self, epochs, optimizer, criterion, batch=True):
        """ Optimizes the parameters of the network to minimize the
            criterion.
            
            epochs: number of optimization steps
            optimizer: torch.optim.Adam(), torch.optim.SGD...
        """
        self.net.train()
        if batch:
            self.create_batches()
            for epoch in range(int(epochs)):
                for local_batch, local_labels in self.batches:
                    optimizer.zero_grad()
                    output = self.net(local_batch).squeeze()
                    loss = criterion(output, local_labels)
                    loss.backward()
                    optimizer.step()
        else:
            for epoch in range(int(epochs)):
                optimizer.zero_grad()
                output = self.net(self.X_train).squeeze()
                loss = criterion(output, self.y_train)
                loss.backward()
                optimizer.step()
        return

    def predict(self):       
        """ Returns the prediction of the neural network for X_test"""       
        self.net.eval()
        self.net.training = True
        self.pred = self.net(self.X_test).squeeze().detach()
        return self.pred

    def plot_results(self, ax=None):      
        """ Plots the training points (scatter plot) as well as the prediction 
        of the network (plot)
        """
        if ax is None:
            ax = plt.subplot()
        ax.scatter(self.X_train.numpy(), self.y_train.numpy(), color='red', marker='x', label="training points")
        ax.plot(self.X_test.numpy(), self.pred.numpy(), color='blue', label="prediction")
        return
