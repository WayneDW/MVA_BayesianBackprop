import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.RL.nets import RLReg
from models.RL.rl_utils import AgentBBNet
from models.bayesbackprop_regression import BayesianLinear


class BayesBackpropRLNet(nn.Module):
    """ Defines a neural-network with one hidden layer with size hidden-size and
        relu activation. Each layer is a BayesianLinear layer defined as above.

        Builds the ELBO function associated with the network:
            KL(q(w) ||p(w)) - E[log(p(D | w))]  (w: all the parameters of the network)
    """

    def __init__(self, hidden_size, dim_context, dim_action_space, prior_parameters, sigma):
        super(BayesBackpropRLNet, self).__init__()
        self.fc1 = BayesianLinear(dim_input=dim_context, dim_output=hidden_size
                                  , prior_parameters=prior_parameters)
        self.fc2 = BayesianLinear(dim_input=hidden_size, dim_output=hidden_size
                                  , prior_parameters=prior_parameters)
        self.fc3 = BayesianLinear(dim_input=hidden_size, dim_output=dim_action_space
                                  , prior_parameters=prior_parameters)
        self.sigma = sigma  # noise associated with the data y = f(x; w) + N(0, self.sigma)

        self.layers = [self.fc1, self.fc3]

    def forward(self, x):
        out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def weights_dist(self):
        """ Return flatten numpy array containing all the weights of the net """
        return np.hstack(list(map(lambda layer: layer.get_weights_mu(), self.layers)))

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

    def sample_elbo(self, x, y, actions, MC_samples, weight):
        """ For a batch x computes weight * E(log(q(w)) - log(p(w))) - E(log(p(D |w)))
            The expected values are computed with a MC scheme (at each step w is sampled
            from q(w)).
        """
        elbo = 0
        log_var_posteriors = 0
        log_priors = 0
        log_likelihoods = 0
        for s in range(MC_samples):
            out = self.forward(x).squeeze()[np.arange(x.shape[0]), actions]

            log_var_posterior = self.log_variational_posterior() * weight
            log_var_posteriors += log_var_posterior

            log_prior = self.log_prior() * weight
            log_priors += log_prior

            log_likelihood = self.log_likelihood(y, out)
            log_likelihoods += log_likelihood

            elbo += log_var_posterior - log_prior - log_likelihood

        return elbo / MC_samples, log_var_posteriors / MC_samples, log_priors / MC_samples, log_likelihoods / MC_samples


class BayesRLReg(RLReg):

    def __init__(self, X_train, y_train, agent, buffer_size=4096, minibatch_size=64, burn_in=500):
        assert type(agent) == AgentBBNet, (type(agent), AgentBBNet)
        super(BayesRLReg, self).__init__(X_train, y_train, agent, buffer_size, minibatch_size, burn_in)

    def aux_optimization_(self, context_inds, actions, rewards):
        loss, log_var_posterior, log_prior, log_likelihood = self.agent.net.sample_elbo(self.X_train[context_inds],
                                                                                        rewards, actions,
                                                                                        self.agent.sample,
                                                                                        weight=1 / self.agent.sample)
        return loss
