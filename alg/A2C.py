import copy
import torch as th
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F

import os
from .model.ac_sharenet import DNNAgent


class ACLearner:
    """
    1. DQN- RNNAgent
    2. train
    """

    def __init__(self, param_set, writer):
        self.obs_shape = param_set['obs_shape'][0]
        self.gamma = param_set['gamma']
        self.learning_rate = param_set['learning_rate']
        self.loss_coeff_entropy = param_set['loss_coeff_entropy']

        self.ac = DNNAgent(param_set)

        self.params = self.ac.parameters()
        self.optimiser = Adam(params=self.params, lr=self.learning_rate)
        self.writer = writer
        self._episode = 0

        self.log_pi_batch = []
        self.value_batch = []


    def new_trajectory(self):
        self.log_pi_batch = []
        self.value_batch = []


    def get_action(self, observation, *arg):
        obs = th.FloatTensor(observation)
        pi, q = self.ac(obs=obs)
        m = Categorical(pi)
        # print(pi)
        if (pi<0).any():
            print(pi)
        action_index = m.sample()

        self.log_pi_batch.append(m.log_prob(action_index))
        self.value_batch.append(q[action_index])
        return int(action_index), pi

    def learn(self, memory):
        batch = memory.get_last_trajectory()

        reward = th.FloatTensor(batch['reward'][0])
        value = th.stack(self.value_batch)
        log_pi = th.stack(self.log_pi_batch)


        # advantage
        advantage = th.zeros_like(reward)
        returns = th.zeros_like(reward)
        deltas = th.zeros_like(reward)
        pre_return = 0
        pre_value = 0
        pre_advantage = 0
        for i in range(advantage.shape[0]-1, -1, -1):
            returns[i] = reward[i] + self.gamma * pre_return
            deltas[i] = reward[i] + self.gamma * pre_value - value[i]
            advantage[i] = deltas[i] + self.gamma * self.lamda * pre_advantage
            pre_return = returns[i]
            pre_value = value[i]
            pre_advantage = advantage[i]


        mask = th.ones_like(value)
        mask[-1] = 0
        next_value = th.cat([value[1:], value[0:1]],dim=-1) * mask

        td_error = reward + self.gamma * next_value.detach() - value
        td_loss = (td_error ** 2).mean()
        entropies = -(log_pi * log_pi.exp()).sum(dim=1, keepdim=True)
        J = - (advantage.detach() * log_pi + entropies).mean()

        loss = J + td_loss

        self.writer.add_scalar('Loss/J', J.item(), self._episode)
        self.writer.add_scalar('Loss/TD_loss', td_loss.item(), self._episode)
        self.writer.add_scalar('Loss/Entropy', entropies.mean().item(), self.step)
        self.writer.add_scalar('Loss/loss', loss.item(), self._episode)


        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, 10)
        self.optimiser.step()

        self._episode += 1




