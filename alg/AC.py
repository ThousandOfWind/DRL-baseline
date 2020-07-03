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
        self.learning_rate = param_set['learning_rate']

        self.pi = DNNAgent(param_set)
        self.V = DNN(param_set)

        self.params = self.pi.parameters()
        self.optimiser = Adam(params=self.params, lr=self.learning_rate)
        self.writer = writer
        self._episode = 0

        self.log_pi_batch = []
        self.b_batch = []


    def new_trajectory(self):
        self.log_pi_batch = []
        self.value_batch = []


    def get_action(self, observation, *arg):
        obs = th.FloatTensor(observation)
        pi, _ = self.pi(obs=obs)
        m = Categorical(pi)
        action_index = m.sample()

        self.log_pi_batch.append(pi)
        self.value_batch.append(self.B(obs=obs))
        return int(action_index), pi

    def learn(self, memory):
        batch = memory.get_last_trajectory()

        reward = th.FloatTensor(batch['reward'][0])
        log_pi = th.stack(self.log_pi_batch)
        value = th.stack(self.value_batch)

        self.value_batch.append(0)
        next_value = th.stack(self.value_batch)[1:]
        G = reward + self.gamma * next_value

        J = - ((G-value).detach() * log_pi).mean()
        value_loss = F.smooth_l1_loss(value, reward).mean()
        loss = J + value_loss

        self.writer.add_scalar('Loss/J', J.item(), self._episode)
        self.writer.add_scalar('Loss/B', value_loss.item(), self._episode)
        self.writer.add_scalar('Loss/loss', loss.item(), self._episode)


        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, 10)
        self.optimiser.step()

        self._episode += 1




