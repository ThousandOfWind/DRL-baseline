import copy
import torch as th
from torch.optim import Adam
from torch.distributions import Categorical

import os
from .model.base import DNNAgent


class REINFORCELearner:
    """
    1. DQN- RNNAgent
    2. train
    """
    def __init__(self, param_set, writer):

        self.obs_shape = param_set['obs_shape'][0]
        self.gamma = param_set['gamma']
        self.learning_rate = param_set['learning_rate']


        self.pi = DNNAgent(param_set)
        self.params = self.pi.parameters()
        self.optimiser = Adam(params=self.params, lr=self.learning_rate)
        self.writer = writer
        self._episode = 0

        self.log_pi_batch = []

    def new_trajectory(self):
        self.log_pi_batch = []


    def get_action(self, observation, *arg):
        obs = th.FloatTensor(observation)
        pi = self.pi(obs=obs)
        m = Categorical(pi)
        action_index = m.sample()

        self.log_pi_batch.append((m.log_prob(action_index)))
        return int(action_index), pi

    def learn(self, memory):
        batch = memory.get_last_trajectory()

        # build G_t
        G = copy.deepcopy(batch['reward'][0])
        for index in range(2, len(G)+1):
            G[-index] += self.gamma * G[-index + 1]
        G = th.FloatTensor(G)

        log_pi = th.stack(self.log_pi_batch)
        J = - (G * log_pi).mean()

        self.writer.add_scalar('Loss/J', J.item(), self._episode )
        self.optimiser.zero_grad()
        J.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, 10)
        self.optimiser.step()

        self._episode += 1





