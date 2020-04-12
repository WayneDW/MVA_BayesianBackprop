import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils import data


# This file proposes an implementation of the Bayes by backprop method (Blundell
# et al. "Weight Uncertainty in Neural Networks") for a regression problem (with
# gaussian noise).
# https://arxiv.org/abs/1505.05424

# The file is attached with a jupyter notebook "regression.ipynb" which illustrates
# the method in a simple case.

class VarPosterior(object):
    """ Defines the variational posterior distribution q(w) for the weights of
        the network.
        
        Here we suppose that q(w) = N(mu , log(1 + exp(rho))).
    """

    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho
        self.gaussian = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        """Returns the variance of the distribution sigma = log(1 + exp(rho))"""
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        """Samples mu + sigma*N(0 , 1)"""
        epsilon = self.gaussian.sample(self.rho.size())
        return self.mu + self.sigma * epsilon

    def log_prob(self, x):
        """Returns the log-distribution of a vector x whose each component is 
           independently distributed according to q(w).
        """
        return torch.sum(-0.5 * torch.log(2 * np.pi * self.sigma ** 2)
                         - 0.5 * (x - self.mu) ** 2 / self.sigma ** 2)


class Prior(object):
    """ Defines the prior distribution p(w) for the weights of the network.
    
        Here we suppose that p(w) = pi*N(0 , sigma1) + (1 - pi)*N(0 , sigma2).
    """

    def __init__(self, sigma1, sigma2, pi):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.pi = pi
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def sample(self):
        """ Samples x*N(0 , sigma1) + (1 - x)*N(0 ,1), where x follows a Bernoulli
            laws of parameter pi.
        """
        x = np.random.binomial(1, self.pi)
        return x * self.gaussian1.sample(torch.Size([1])) + (1 - x) * self.gaussian2.sample(torch.Size([1]))

    def log_prob(self, x):
        """Returns the log-distribution of a vector x whose each component is 
           independently distributed according to p(w).
           
           To deal with overflows we compute:
               log(pi) + log(Gauss1) + log(1 + (1 - pi)/pi * Gauss2/Gauss1)
        """
        function = lambda x: x * np.exp(-x ** 2)
        return torch.sum(np.log(self.pi) + self.gaussian1.log_prob(x)
                         + np.log1p(((1 - self.pi) / self.pi) * function(self.sigma1 / self.sigma2)))


class BayesianLinear(nn.Module):
    """ Defines a linear layer for a neural network.
    
        The weights w are distributed according to the posterior q(w ; w_mu , w_rho)
        The biases b are also distributed according to the posterior q(b ; b_mu , b_rho)
        
        w and b are associated with priors p(w ; sigma1,sigma2,pi) and p(b ; sigma1,sigma2,pi)
    """

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
        """ Samples a couple (w , b) with the variational posteriors, saves the
            log-likelihoods of this sample and returns the output of the layer
            computed with these samples.
        """
        w = self.w.sample()
        b = self.b.sample()

        self.log_prior = self.w_prior.log_prob(w) + self.b_prior.log_prob(b)
        self.log_variational_posterior = self.w.log_prob(w) + self.b.log_prob(b)

        return F.linear(x, w, b)

    def get_weights_mu(self):
        """ Auxiliary function used to get the weight distribution of a net """
        return np.hstack([self.w_mu.detach().numpy().flatten(), self.b_mu.detach().numpy().flatten()])


class BayesBackpropNet(nn.Module):
    """ Defines a neural-network with one hidden layer with size hidden-size and
        relu activation. Each layer is a BayesianLinear layer defined as above.
        
        Builds the ELBO function associated with the network:
            KL(q(w) ||p(w)) - E[log(p(D | w))]  (w: all the parameters of the network)
    """

    def __init__(self, hidden_size, dim_input, dim_output, prior_parameters, sigma):
        super(BayesBackpropNet, self).__init__()
        self.fc1 = BayesianLinear(dim_input=dim_input, dim_output=hidden_size
                                  , prior_parameters=prior_parameters)
        self.fc2 = BayesianLinear(dim_input=hidden_size, dim_output=dim_output
                                  , prior_parameters=prior_parameters)

        self.sigma = sigma  # noise associated with the data y = f(x; w) + N(0, self.sigma)
        self.layers = [self.fc1, self.fc2]

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

    def log_prior(self):
        """ Computes log(p(w)) """
        return sum(map(lambda fc: fc.log_prior, self.layers))

    def log_variational_posterior(self):
        """ Computes log(q(w|D)) """
        return sum(map(lambda fc: fc.log_variational_posterior, self.layers))

    def log_likelihood(self, y, output):
        """ Computes log(p(D|w))
        
            Rmk: y_i = f(x_i ; w) + epsilon (epsilon ~ N(0 , self.sigma)) 
                 So we have p(y_i | x_i , w) = N(f(x_i ; w) , self.sigma)
        """
        return torch.sum(-0.5 * np.log(2 * np.pi * self.sigma ** 2) - 0.5 * (y - output) ** 2 / self.sigma ** 2)

    def sample_elbo(self, x, y, MC_samples, weight):
        """ For a batch x computes weight * E(log(q(w)) - log(p(w))) - E(log(p(D |w)))
            The expected values are computed with a MC scheme (at each step w is sampled
            from q(w)).
        """
        elbo = 0
        log_var_posteriors = 0
        log_priors = 0
        log_likelihoods = 0
        for s in range(MC_samples):
            out = self.forward(x).squeeze()

            log_var_posterior = self.log_variational_posterior() * weight
            log_var_posteriors += log_var_posterior

            log_prior = self.log_prior() * weight
            log_priors += log_prior

            log_likelihood = self.log_likelihood(y, out)
            log_likelihoods += log_likelihood

            elbo += log_var_posterior - log_prior - log_likelihood

        return elbo / MC_samples, log_var_posteriors / MC_samples, log_priors / MC_samples, log_likelihoods / MC_samples

    def weights_dist(self):
        """ Return flatten numpy array containing all the weights of the net """
        return np.hstack(list(map(lambda layer: layer.get_weights_mu(), self.layers)))


class BayesBackpropReg(object):
    """Defines the regression model for the Bayes by backprop model.
       The training set (X_train , y_train) and the test set X_test are given.
    """

    def __init__(self, X_train, y_train, X_test, net, batch_size):
        self.net = net
        self.batch_size = batch_size
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.pred, self.pred_mean, self.pred_std = None, None, None
        self.batches = self.create_batches()
        self.nb_batches = len(self.batches)
        self.writer = SummaryWriter()  # to get learning curves: tensorboard --logdir=runs (in console)
        self.step = 0
        self.epoch = 0
        self.execution_time = 0

    def create_batches(self):
        torch_train_dataset = data.TensorDataset(self.X_train, self.y_train)
        return data.DataLoader(torch_train_dataset, batch_size=self.batch_size)

    def train(self, epochs, optimizer, MC_samples, weights='uniform'):
        """ Optimizes the parameters of the network to minimize the
            ELBO loss.            
            At each optimization step and for each mini-batch: estimates the 
            ELBO loss with a MC algorithm (MC_samples steps).
            
            epochs: number of optimization steps
            optimizer: torch.optim.Adam(), torch.optim.SGD...
            weights: weighting strategy for the KL divergence within the ELBO loss
                     either "uniform" or "geometric".            
        """
        self.net.train()
        for _ in range(int(epochs)):
            i = 0
            t = time.time()
            self.epoch += 1
            elbos, log_var_posteriors, log_priors, log_likelihoods = 0, 0, 0, 0
            for local_batch, local_labels in self.batches:
                i += 1
                self.step += 1
                optimizer.zero_grad()
                if weights == 'uniform':
                    weight = 1 / len(self.batches)
                elif weights == 'geometric':
                    weight = 2 ** (self.nb_batches - i) / (2 ** self.nb_batches - 1)
                else:
                    raise ValueError("wrong argument for @weight")
                loss, log_var_posterior, log_prior, log_likelihood = self.net.sample_elbo(local_batch, local_labels,
                                                                                          MC_samples, weight=weight)
                loss.backward()
                optimizer.step()
                elbos += loss
                log_var_posteriors += log_var_posterior
                log_priors += log_prior
                log_likelihoods += log_likelihood
            self.execution_time += time.time() - t

            self.writer.add_scalar('loss/elbo', elbos, self.epoch)
            self.writer.add_scalar('loss/complexity_cost', log_var_posteriors - log_priors, self.epoch)
            self.writer.add_scalar('loss/negative log-likelihood', - log_likelihoods, self.epoch)
            self.writer.add_scalar('execution_time', self.execution_time, self.epoch)
            if self.epoch % 100 == 0:
                print("Epoch: %4d/%4d, elbo loss = %8.1f, KL = %8.1f, -log-likelihood = %6.1f" %
                      (self.epoch, epochs, elbos, log_var_posteriors - log_priors, - log_likelihoods))
        return

    def predict(self, samples):
        """ Runs a Monte Carlo algorithm for the prediction of the network.
            Aggregates all the predictions and return the mean and the standard
            deviation.
        """
        self.net.eval()
        self.pred = torch.zeros((self.X_test.shape[0], self.y_train.unsqueeze(dim=1).shape[1], samples))
        for s in range(samples):
            self.pred[:, :, s] = self.net(self.X_test).detach()

        self.pred_mean = torch.mean(self.pred, dim=2).squeeze()
        self.pred_std = torch.std(self.pred, dim=2).squeeze()

        return self.pred_mean, self.pred_std

    def plot_results(self, ax=None):
        """ Plots the training points, the mean prediction of the network for
            X_test, and the confidence intervals for std, 2*std and 3*std.
        """
        if ax is None:
            ax = plt.subplot()

        X_test = self.X_test.squeeze().numpy()
        y_pred = self.pred_mean.squeeze().numpy()
        std_pred = self.pred_std.squeeze().numpy()

        ax.fill_between(X_test, y_pred - std_pred * 3, y_pred + std_pred * 3, color='mistyrose', label='3 std. int.')
        ax.fill_between(X_test, y_pred - std_pred * 2, y_pred + std_pred * 2, color='lightcoral', label='2 std. int.')
        ax.fill_between(X_test, y_pred - std_pred, y_pred + std_pred, color='indianred', label='1 std. int.')

        ax.scatter(self.X_train.numpy(), self.y_train.numpy(), color='red', marker='x', label="training points")
        ax.plot(X_test, y_pred, color='blue', label="prediction")
        return
