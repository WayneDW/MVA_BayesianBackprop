import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.nn import PyroSample

#from abc import ABCMeta, abstractmethod

class Prior(object): #, metaclass=ABCMeta):
    
    """ Defines the prior distribution p(w) for the weights of the network.
    
        Here we suppose that p(w) = pi*N(0 , sigma1) + (1 - pi)*N(0 , sigma2).
    """

    def __init__(self, sigma1, sigma2, pi , dim):
        self.dim = dim
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.pi = pi
        self.gaussian1 = dist.Normal(0, sigma1).expand(dim).to_event(len(dim))
        self.gaussian2 = dist.Normal(0, sigma2).expand(dim).to_event(len(dim))
    
    def __call__(self):
        return self.sample()
        
#    @abstractmethod
    def sample(self):
        """ Samples x*N(0 , sigma1) + (1 - x)*N(0 ,1), where x follows a Bernoulli
            laws of parameter pi.
        """
        x = np.random.binomial(1, self.pi)
        return x * self.gaussian1.sample(torch.Size([1])) + (1 - x) * self.gaussian2.sample(torch.Size([1]))
    
#    @abstractmethod
    def log_prob(self, x):
        """Returns the log-distribution of a vector x whose each component is 
           independently distributed according to p(w).
           
           To deal with overflows we compute:
               log(pi) + log(Gauss1) + log(1 + (1 - pi)/pi * Gauss2/Gauss1)
        """  
        function = lambda x: x*np.exp(-x**2)
        return torch.sum(np.log(self.pi) + self.gaussian1.log_prob(x) 
                            + np.log1p( ((1 - self.pi) / self.pi)*function(self.sigma1/self.sigma2)))   
    

class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features , hidden_size , prior_parameters):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](in_features, hidden_size)
        self.fc2 = PyroModule[nn.Linear](hidden_size, out_features)
        
        # self.fc1.weight = PyroSample(Prior(prior_parameters['sigma1']
        #                  , prior_parameters['sigma2'], prior_parameters['pi'], dim = [hidden_size , in_features]))
        # self.fc1.bias = PyroSample(Prior(prior_parameters['sigma1']
        #                                  , prior_parameters['sigma2'], prior_parameters['pi'],dim = [hidden_size]))
        # self.fc2.weight = PyroSample(Prior(prior_parameters['sigma1']
        #                 , prior_parameters['sigma2'], prior_parameters['pi'],dim = [out_features , hidden_size]))
        # self.fc2.bias = PyroSample(Prior(prior_parameters['sigma1']
        #                                  ,prior_parameters['sigma2'], prior_parameters['pi'],dim = [out_features]))

        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_size, in_features]).to_event(2))
        self.fc2.weight =  PyroSample(dist.Normal(0., 1.).expand([out_features,hidden_size]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_size]).to_event(1))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([out_features]).to_event(1))
        
    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 1.))
        mean = self.fc2(F.relu(self.fc1(x))).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean
    

from pyro.infer import SVI, Trace_ELBO  
from pyro.infer import Predictive

class PyroReg_SVI(object):
    
    def __init__(self , X_train , y_train , X_test , model , guide):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.model = model
        self.guide = guide
        self.pred = {}
    
    def train(self, epochs , optimizer):
        
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        pyro.clear_param_store()
        
        for epoch in range(epochs):
            loss = svi.step(self.X_train, self.y_train)
            if epoch % 100 == 0:
                print("[iteration %04d] loss: %.4f" % (epoch, loss))
        return
    
    def predict(self, n_samples):
        predictive = Predictive(self.model, guide = self.guide, num_samples= n_samples 
                                , return_sites=("linear.weight", "obs", "_RETURN"))
        samples = predictive(self.X_test)
        for k, v in samples.items():
            self.pred[k] = {"mean": torch.mean(v, 0),
                            "std": torch.std(v, 0),
                           }
        return
    
    def plot_results(self, ax=None):
        if ax is None:
            ax = plt.subplot()

        X_test = self.X_test.squeeze().numpy()
        y_pred = self.pred["_RETURN"]["mean"].detach().squeeze().numpy()
        std_pred = self.pred["_RETURN"]["std"].detach().numpy()

        ax.fill_between(X_test, y_pred - std_pred * 3, y_pred + std_pred * 3, color='mistyrose', label='3 std. int.')
        ax.fill_between(X_test, y_pred - std_pred * 2, y_pred + std_pred * 2, color='lightcoral', label='2 std. int.')
        ax.fill_between(X_test, y_pred - std_pred, y_pred + std_pred, color='indianred', label='1 std. int.')

        ax.scatter(self.X_train.numpy(), self.y_train.numpy(), color='red', marker='x', label="training points")
        ax.plot(X_test, y_pred, color='blue', label="prediction")
        return