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

    def get_action(self, observation, *arg):
        obs = th.FloatTensor(observation)
        pi, _ = self.pi(obs=obs)
        action_index = th.multinomial(pi, 1).squeeze(-1)
        return int(action_index), pi

    def learn(self, memory):
        batch = memory.get_last_trajectory()

        # build G_t
        G = copy.deepcopy(batch['reward'][0])
        for index in range(2, len(G)+1):
            G[-index] += self.gamma * G[-index + 1]

        obs = th.FloatTensor(batch['observation'][0])
        action_index = th.LongTensor(batch['action_index'][0])
        pi, log_pi = self.pi(obs=obs)
        log_pi = th.gather(log_pi, dim=1, index=action_index.unsqueeze(-1)).squeeze(-1)
        G = th.FloatTensor(G)

        J = - (G * log_pi).mean()
        self.writer.add_scalar('Loss/J', J.item(), self._episode )
        self.optimiser.zero_grad()
        J.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, 10)
        self.optimiser.step()

        self._episode += 1





