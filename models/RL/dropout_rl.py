import numpy as np
import torch

from models.RL.rl_utils import RLReg
from models.regression.dropout_regression import DropoutNet


class DropoutRLNet(DropoutNet):
    """ Essentially similar to a DropoutNet defined in dropout_regression.py """

    def __init__(self, hidden_size, dim_context, dim_action_space, p):
        super(DropoutRLNet, self).__init__(hidden_size, dim_input=dim_context, dim_output=dim_action_space, p=p)


class DropoutRLReg(RLReg):
    """ Class for training an AgentB """

    def __init__(self, X_train, y_train, agent, buffer_size=4096, minibatch_size=64, burn_in=500,
                 criterion=torch.nn.MSELoss()):
        super(DropoutRLReg, self).__init__(X_train, y_train, agent, buffer_size, minibatch_size, burn_in)
        self.criterion = criterion

    def get_loss_(self, context_inds, actions, rewards):
        rewards_preds = self.agent.evaluate(
            torch.repeat_interleave(self.X_train[context_inds], self.agent.sample, dim=1))
        reward_preds = rewards_preds[np.arange(self.minibatch_size), actions]
        loss = self.criterion(reward_preds, rewards)
        return loss
