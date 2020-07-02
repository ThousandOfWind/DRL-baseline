import copy
import torch as th
from torch.optim import Adam


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

    def get_action(self, observation):
        obs = th.FloatTensor(observation)
        pi, _ = self.pi(obs=obs)
        action_index = th.multinomial(pi, 1).squeeze(-1)
        return action_index

    def learn(self, memory):
        batch = memory.get_last_trajectory()

        # build G_t
        G = copy.deepcopy(batch['reward'])
        for index in range(2, len(G)+1):
            G[-index] += self.gamma * G[-index + 1]

        obs = th.FloatTensor(batch['observation'])
        action_index = th.LongTensor(batch['action_index'])
        pi, log_pi = self.pi(obs=obs)
        log_pi = th.gather(log_pi[:, :-1], dim=1, index=action_index).squeeze(-1)
        G = th.FloatTensor(G)

        loss = - (G * log_pi).mean()
        self.writer.add_scalar('Loss/TD_loss', loss.item(), self._episode )
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, 10)
        self.optimiser.step()

        self._episode += 1





