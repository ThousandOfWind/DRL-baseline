import copy
import torch as th
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F

import os
from .model.base import DNNAgent
from .model.critic import DNN


class ACLearner:
    """
    1. DQN- RNNAgent
    2. train
    """

    def __init__(self, param_set, writer):
        self.obs_shape = param_set['obs_shape'][0]
        self.gamma = param_set['gamma']

        self.pi = DNNAgent(param_set)
        self.V = DNN(param_set)

        self.params = [
                        {'params': self.pi.parameters(), 'lr':param_set['pi_learning_rate']},
                        {'params': self.V.parameters(), 'lr':param_set['V_learning_rate']}
                    ]
        self.optimiser = Adam(self.params)

        self.writer = writer
        self._episode = 0



    def new_trajectory(self):
        self.I = 1


    def get_action(self, observation, *arg):
        obs = th.FloatTensor(observation)
        pi = self.pi(obs=obs)
        m = Categorical(pi)
        action_index = m.sample()
        self.log_pi = m.log_prob(action_index)
        return int(action_index), pi

    def learn(self, memory):
        batch = memory.get_last_trajectory()

        obs = th.FloatTensor(batch['observation'])
        next_obs = th.FloatTensor(batch['next_obs'])

        value = self.V(obs).squeeze(0)
        next_value = self.V(next_obs).squeeze()

        td_error = batch['reward'][0] + batch['done'][0] * self.gamma * next_value.detach() - value

        J = - ((self.I * td_error).detach() * self.log_pi)
        self.I *= self.gamma
        value_loss = (td_error ** 2)
        loss = J + value_loss

        self.writer.add_scalar('Loss/J', J.item(), self._episode)
        self.writer.add_scalar('Loss/B', value_loss.item(), self._episode)
        self.writer.add_scalar('Loss/loss', loss.item(), self._episode)

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.pi.parameters(), 10)
        grad_norm = th.nn.utils.clip_grad_norm_(self.V.parameters(), 10)
        self.optimiser.step()

        self._episode += 1




