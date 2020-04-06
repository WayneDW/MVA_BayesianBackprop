import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


class DropoutNet(nn.Module):

    def __init__(self, hidden_size, dim_input, dim_output, p):
        super().__init__()
        self.fc1 = nn.Linear(dim_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, dim_output)
        self.p = p

    def forward(self, x):
        return self.fc2(F.dropout(F.relu(self.fc1(x)), self.p))

    def weights_dist(self):
        """ Return flatten numpy array containing all the weights of the net """
        return np.hstack([self.fc1.weight.data.numpy().flatten(), self.fc2.weight.data.numpy().flatten()])


class DropoutReg(object):

    def __init__(self, X_train, y_train, X_test, net, batch_size):
        self.net = net
        self.batch_size = batch_size
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.pred, self.pred_mean, self.pred_std = None, None, None
        self.batches = self.create_batches()

    def create_batches(self):
        torch_train_dataset = data.TensorDataset(self.X_train, self.y_train)
        return data.DataLoader(torch_train_dataset, batch_size=self.batch_size)

    def train(self, epochs, optimizer, criterion):
        self.net.train()
        for epoch in range(int(epochs)):
            for local_batch, local_labels in self.batches:
                optimizer.zero_grad()
                output = self.net(local_batch).squeeze()
                loss = criterion(output, local_labels)
                loss.backward()
                optimizer.step()
        return

    def predict(self, samples):

        self.net.eval()
        self.net.training = True
        self.pred = torch.zeros((self.X_test.shape[0], self.y_train.unsqueeze(dim=1).shape[1], samples))
        for s in range(samples):
            self.pred[:, :, s] = self.net(self.X_test).detach()

        self.pred_mean = torch.mean(self.pred, dim=2).squeeze()
        self.pred_std = torch.std(self.pred, dim=2).squeeze()

        return self.pred_mean, self.pred_std

    def plot_results(self, ax=None):
        if ax is None:
            ax = plt.subplot()
        X_test = self.X_test.squeeze().numpy()
        y_pred = self.pred_mean.squeeze().numpy()
        std_pred = self.pred_std.squeeze().numpy()

        ax.fill_between(X_test, y_pred - std_pred, y_pred + std_pred, color='indianred', label='1 std. int.')
        ax.fill_between(X_test, y_pred - std_pred * 2, y_pred - std_pred, color='lightcoral')
        ax.fill_between(X_test, y_pred + std_pred * 1, y_pred + std_pred * 2, color='lightcoral', label='2 std. int.')
        ax.fill_between(X_test, y_pred - std_pred * 3, y_pred - std_pred * 2, color='mistyrose')
        ax.fill_between(X_test, y_pred + std_pred * 2, y_pred + std_pred * 3, color='mistyrose', label='3 std. int.')

        ax.scatter(self.X_train.numpy(), self.y_train.numpy(), color='red', marker='x', label="trainig points")
        ax.plot(X_test, y_pred, color='blue', label="prediction")
        return
