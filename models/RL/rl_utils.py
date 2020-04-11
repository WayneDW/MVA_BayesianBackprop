from collections import deque

import numpy as np


class Agent(object):

    def __init__(self):
        pass

    def act(self, context):
        raise NotImplementedError()

    def train(self):
        pass

    def eval(self):
        pass


class AgentDN(Agent):
    """ Deep Net Agent """

    def __init__(self, net):
        super(AgentDN, self).__init__()
        self.net = net

    def evaluate(self, context):
        return self.net(context).squeeze()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()


class AgentGreedy(AgentDN):

    def __init__(self, net, epsilon):
        super(AgentGreedy, self).__init__(net)
        self.epsilon = epsilon

    def act(self, context):
        predictions = self.evaluate(context)  # predict potential rewards
        filtr = np.random.random(size=context.shape[0]) > self.epsilon
        action = predictions.argmax(axis=-1).squeeze() * filtr + np.random.randint(low=0, high=2,
                                                                                   size=context.shape[0]) * (1 - filtr)
        return action


class AgentBBNet(AgentDN):

    def __init__(self, bayes_net, sample=2):
        """ Initialize Bayesian net Agent
        :param sample: int
            number of samples to estimate rewards
        """
        super(AgentBBNet, self).__init__(bayes_net)
        self.sample = sample

    def act(self, context):
        predictions = self.evaluate(context)
        for _ in range(self.sample - 1):
            predictions += self.evaluate(context)
        return predictions.argmax(axis=-1).squeeze()


class Env(object):

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
            reward = 0
        elif self.is_edible:
            reward = 5
        else:
            reward = np.random.choice([-35, 5])
        return self.reset(), reward


class ReplayBuffer(object):
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
