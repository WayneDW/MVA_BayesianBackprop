import time
from collections import deque

import numpy as np
import torch

"""
In this file are created the classes related to the RL machinery (environment, agent, ...)
"""


class Agent(object):
    """ Agent abstract class """

    def __init__(self):
        pass

    def act(self, context):
        raise NotImplementedError()

    def train(self):
        pass

    def eval(self):
        pass


class AgentDN(Agent):
    """ Deep Net Agent abstract class """

    def __init__(self, net):
        super(AgentDN, self).__init__()
        self.net = net

    def evaluate(self, context):
        if context.ndim > 2:
            return self.net(context).mean(axis=1)
        else:
            return self.net(context).squeeze()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()


class AgentGreedy(AgentDN):
    """ Greedy Agent: chooses best action with probability (1 - epsilon), random one with probability epsilon """

    def __init__(self, net, epsilon):
        super(AgentGreedy, self).__init__(net)
        self.epsilon = epsilon

    def act(self, context):
        predictions = self.evaluate(context)  # predict potential rewards
        filtr = np.random.random(size=context.shape[0]) > self.epsilon
        action = predictions.argmax(axis=-1).squeeze() * filtr + np.random.randint(low=0, high=2,
                                                                                   size=context.shape[0]) * (1 - filtr)
        return action


class AgentBayesNet(AgentDN):

    def __init__(self, bayes_net, sample):
        """ Initialize Bayesian net Agent
        :param sample: int
            number of samples to estimate rewards
        """
        super(AgentBayesNet, self).__init__(bayes_net)
        self.sample = sample

    def act(self, context):
        if context.ndim < 3:
            context = context.unsqueeze(0)
        predictions = self.evaluate(torch.repeat_interleave(context, self.sample, dim=1))
        return predictions.argmax(axis=-1).squeeze()


class AgentDropout(AgentBayesNet):
    """ Dropout Agent (seen as Bayesian one) """

    def __init__(self, net, sample=2):
        super(AgentDropout, self).__init__(net, sample)


class AgentBayesBackprop(AgentBayesNet):
    """ Bayes by Backprop Agent """

    def __init__(self, net, sample=2):
        super(AgentBayesBackprop, self).__init__(net, sample)


class EnvMushroom(object):
    """ Environment based on mushroom dataset
    Mushrooms have a list of features and are either edible (0) or poisonous (1)
    Remark: environment state is only charactrized by an integer referring to a row in the mushroom dataset
    """

    def __init__(self, context_inds, classes):
        self.context_inds = context_inds
        self.classes = classes
        self.context_ind = None

    @property
    def is_edible(self):
        return self.classes[self.context_ind] == 0

    @property
    def oracle(self):
        return 5 if self.is_edible else 0

    def reset(self):
        self.context_ind = np.random.randint(0, len(self.context_inds))
        return self.context_ind

    def step(self, action):
        assert action in [0, 1]
        if action == 0:
            reward = 0  # do not eat -> reward is 0
        elif self.is_edible:
            # eat an edible mushroom
            reward = 5
        else:
            # eat a poisonous mushroom
            reward = np.random.choice([-35, 5])
        return self.reset(), reward


class ReplayBuffer(object):
    """ Replay buffer from which minibatches of [contexts, actions, rewards] are drawn for training """

    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)

    def sample(self, batch_size):
        assert batch_size < len(self)
        indices = np.random.choice(len(self), batch_size, replace=False)  # select indices of selected elements
        context_inds, actions, rewards = zip(*[self.buffer[idx] for idx in indices])
        return context_inds, actions, rewards

    def add(self, context_ind, action, reward):
        self.buffer.append([context_ind, action, reward])

    def __len__(self):
        return len(self.buffer)


class RLReg(object):
    """ Abstract class for training an AgentDN """

    def __init__(self, X_train, y_train, agent, buffer_size=4096, minibatch_size=64, burn_in=256):
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

            if self.episode < self.burn_in:  # random action
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
                loss = self.get_loss_(context_inds, actions, rewards)
                loss.backward()
                optimizer.step()
                self.training_time += time.time() - current_time
                if self.episode % 500 == 0:
                    print(f"epoch: {self.episode:5d} / {episodes} | loss: {loss:11.1f} | regret: {self.regret:5d} |" +
                          f" training time: {time.strftime('%H:%M:%S', time.gmtime(self.training_time))}")

        return

    def get_loss_(self, context_inds, actions, rewards):
        """ Function called in the main training loop to get the loss to minimize """
        raise NotImplementedError()
