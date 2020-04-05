# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


class DeterministicNet(nn.Module):

    def __init__(self, hidden_size, dim_input, dim_output):
        super().__init__()
        self.fc1 = nn.Linear(dim_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, dim_output)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class DeterministicReg():

    def __init__(self, X_train, y_train, X_test, net, batch_size=None):
        self.net = net
        self.batch_size = batch_size
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.pred = None

    def create_batches(self):
        torch_train_dataset = data.TensorDataset(self.X_train, self.y_train)
        self.batches = data.DataLoader(torch_train_dataset, batch_size=self.batch_size)

    def train(self, epochs, optimizer, criterion, batch=True):

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
        self.net.eval()
        self.net.training = True
        self.pred = self.net(self.X_test).squeeze().detach()
        return self.pred

    def plot_results(self):
        plt.scatter(self.X_train.numpy(), self.y_train.numpy(), color='red', marker='x', label="training points")
        plt.plot(self.X_test.numpy(), self.pred.numpy(), color='blue', label="prediction")
        return
