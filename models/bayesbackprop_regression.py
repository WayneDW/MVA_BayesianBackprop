# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:07:16 2020

@author: nicol
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


class VarPosterior(object):

    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho
        self.sigma = torch.log(1 + torch.exp(self.rho))
        self.gaussian = torch.distributions.Normal(0, 1)

    def sample(self):
        epsilon = self.gaussian.sample(self.rho.size())
        return self.mu + self.sigma * epsilon

    def log_prob(self, x):
        return torch.sum(-0.5 * torch.log(2 * np.pi * self.sigma ** 2) - (x - self.mu) ** 2 / self.sigma ** 2)


class Prior(object):

    def __init__(self, sigma1, sigma2, pi):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.pi = pi
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def sample(self):
        x = np.random.binomial(1, self.pi)
        return x * self.gaussian1.sample(torch.Size([1])) + (1 - x) * self.gaussian2.sample(torch.Size([1]))

    def log_prob(self, x):
        return (torch.log(self.pi * self.gaussian1.log_prob(x).exp()
                          + (1 - self.pi) * self.gaussian2.log_prob(x).exp())).sum()


class BayesianLinear(nn.Module):

    def __init__(self, dim_input, dim_output, prior_parameters):
        super(BayesianLinear, self).__init__()

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.prior_parameters = prior_parameters

        self.w_mu = nn.Parameter(torch.Tensor(dim_output, dim_input).normal_(0, 1))
        self.w_rho = nn.Parameter(torch.Tensor(dim_output, dim_input).normal_(0, 1))
        self.w = VarPosterior(self.w_mu, self.w_rho)
        self.w_prior = Prior(prior_parameters['sigma1'], prior_parameters['sigma2'], prior_parameters['pi'])

        self.b_mu = nn.Parameter(torch.Tensor(dim_output).normal_(0, 1))
        self.b_rho = nn.Parameter(torch.Tensor(dim_output).normal_(0, 1))
        self.b = VarPosterior(self.b_mu, self.b_rho)
        self.b_prior = Prior(prior_parameters['sigma1'], prior_parameters['sigma2'], prior_parameters['pi'])

        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        w = self.w.sample()
        b = self.b.sample()

        self.log_prior = self.w_prior.log_prob(w) + self.b_prior.log_prob(b)
        self.log_variational_posterior = self.w.log_prob(w) + self.b.log_prob(b)

        return x @ torch.t(w) + b


class BayesBackpropNet(nn.Module):

    def __init__(self, hidden_size, dim_input, dim_output, prior_parameters, sigma):
        super().__init__()
        self.fc1 = BayesianLinear(dim_input=dim_input, dim_output=hidden_size
                                  , prior_parameters=prior_parameters)
        self.fc2 = BayesianLinear(dim_input=hidden_size, dim_output=dim_output
                                  , prior_parameters=prior_parameters)

        self.sigma = sigma  # noise associated with the data y = f(x ; w) + N(0 , self.sigma)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

    def log_prior(self):
        """ Compute log(p(w)) """
        return self.fc1.log_prior + self.fc2.log_prior

    def log_variational_posterior(self):
        """ Compute log(q(w|D)) """

        return self.fc1.log_variational_posterior + self.fc2.log_variational_posterior

    def log_likelihood(self, y, output):
        """ Compute log(p(D|w))
        
            Rmk: y_i = f(x_i ; w) + epsilon (epsilon ~ N(0 , self.sigma)) 
                 So we have p(y_i | x_i , w) = N(f(x_i ,; w) , self.sigma)
        """
        return torch.sum(-0.5 * np.log(2 * np.pi * self.sigma ** 2) - (y - output) ** 2 / self.sigma ** 2)

    def sample_elbo(self, x, y, MC_samples, weight):
        """ For a batch x compute weight*E(log(q(w|D)) - log(p(w))) - E(log(p(D |w)))
            The expected values are computed with a MC scheme (at each step w is sampled
            from q(w | D))
        """
        elbo = 0
        for s in range(MC_samples):
            out = self.forward(x)
            elbo += (1 / MC_samples) * (weight * (self.log_variational_posterior() - self.log_prior())
                                        - self.log_likelihood(y, out)
                                        )
        return elbo


class BayesBackpropReg(object):

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

    def train(self, epochs, optimizer, MC_samples, weights='uniform', pi=None):
        self.net.train()
        for epoch in range(int(epochs)):
            for local_batch, local_labels in self.batches:
                optimizer.zero_grad()
                if weights == 'uniform':
                    loss = self.net.sample_elbo(local_batch, local_labels, MC_samples, weight=len(self.batches))
                loss.backward()
                optimizer.step()
        return

    def predict(self, samples):
        self.net.eval()
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

        ax.fill_between(X_test, y_pred - std_pred * 3, y_pred + std_pred * 3, color='mistyrose', label='3 std. int.')
        ax.fill_between(X_test, y_pred - std_pred * 2, y_pred + std_pred * 2, color='lightcoral', label='2 std. int.')
        ax.fill_between(X_test, y_pred - std_pred, y_pred + std_pred, color='indianred', label='1 std. int.')

        ax.scatter(self.X_train.numpy(), self.y_train.numpy(), color='red', marker='x', label="trainig points")
        ax.plot(X_test, y_pred, color='blue', label="prediction")
        return
