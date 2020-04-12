import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.RL.rl_utils import ReplayBuffer, EnvMushroom


class DeterministicRLNet(nn.Module):
    """ Defines a neural network with two hidden layers of size hidden_size. A
        relu activation is applied after the hidden layer.
    """

    def __init__(self, hidden_size, dim_context, dim_action_space):
        super().__init__()
        self.fc1 = nn.Linear(dim_context, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, dim_action_space)
        self.layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(out))
        return self.fc3(out)

    def weights_dist(self):
        """ Return flatten numpy array containing all the weights of the net """
        return np.hstack(list(map(lambda layer: layer.weight.data.numpy().flatten(), self.layers)))


class RLReg(object):

    def __init__(self, X_train, y_train, agent, buffer_size=4096, minibatch_size=64, burn_in=500):
        self.agent = agent
        self.buffer = ReplayBuffer(buffer_size)
        self.X_train = X_train
        self.env = EnvMushroom(np.arange(len(X_train)), y_train)
        self.regret = 0
        self.episode = 0
        self.burn_in = burn_in
        self.minibatch_size = minibatch_size
        self.context_ind = self.env.reset()
        self.hist = {'regret': [self.regret]}
        self.training_time = 0

    def train(self, episodes, optimizer):
        self.agent.train()
        for _ in range(episodes):
            current_time = time.time()
            self.episode += 1

            oracle = self.env.oracle

            if self.episode < self.burn_in: # random action
                action = np.random.randint(0, 2)
            else:
                action = self.agent.act(self.X_train[self.context_ind]).item()

            next_context_ind, reward = self.env.step(action)
            self.buffer.add(self.context_ind, action, reward)
            self.context_ind = next_context_ind
            self.regret += oracle - reward
            self.hist['regret'].append(self.regret)
            if len(self.buffer) > self.burn_in:
                optimizer.zero_grad()
                context_inds, actions, rewards = self.buffer.sample(self.minibatch_size)
                context_inds, actions, rewards = np.array(context_inds), np.array(actions), torch.tensor(
                    np.array(rewards, dtype=float)).float()
                loss = self.aux_optimization_(context_inds, actions, rewards)
                loss.backward()
                optimizer.step()
                self.training_time += time.time() - current_time
                if self.episode % 500 == 0:
                    print(f"epoch: {self.episode:5d} / {episodes} | loss: {loss:11.1f} | regret: {self.regret:5d} |" +
                          f" training time: {time.strftime('%H:%M:%S', time.gmtime(self.training_time))}")

        return

    def predict(self, contexts):
        self.agent.eval()
        return self.agent.act(contexts).squeeze().detach()

    def aux_optimization_(self, context_inds, actions, rewards):
        raise NotImplementedError()


class DeterministicRLReg(RLReg):

    def __init__(self, X_train, y_train, agent, criterion=torch.nn.MSELoss(), buffer_size=4096, minibatch_size=64,
                 burn_in=500):
        super(DeterministicRLReg, self).__init__(X_train, y_train, agent, buffer_size, minibatch_size, burn_in)
        self.criterion = criterion

    def aux_optimization_(self, context_inds, actions, rewards):
        rewards_preds = self.agent.evaluate(self.X_train[context_inds])
        reward_preds = rewards_preds[np.arange(self.minibatch_size), actions]
        loss = self.criterion(reward_preds, rewards)
        return loss
