"""
https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py
"""

import copy
import torch as th
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

import os
from .model.actor_critic import Policy


class ACLearner:
    """
    1. DQN- RNNAgent
    2. train
    """

    def __init__(self, param_set, writer):
        self.obs_shape = param_set['obs_shape'][0]
        self.gamma = param_set['gamma']
        self.learning_rate = param_set['learning_rate']
        self.clip = param_set['clip']

        self.ac = Policy(param_set)

        self.params = self.ac.parameters()
        self.optimiser = Adam(params=self.params, lr=self.learning_rate)
        self.writer = writer
        self._episode = 0

        self.ppo_epoch = param_set['ppo_epoch']
        self.minibatch_size = param_set['minibatch_size']
        self.batch_size = param_set['mamory_size']





    def get_action(self, observation, *arg):
        obs = th.FloatTensor(observation)
        pi, value, _ = self.ac.select_action(obs=obs)
        m = Categorical(pi)
        action_index = m.sample()

        return int(action_index), pi, value

    def learn(self, memory):
        batch = memory.get_current_trajectory()

        obs = th.FloatTensor(batch['observation'])
        action_index = th.LongTensor(batch['action_index'])
        next_obs = th.FloatTensor(batch['next_obs'])
        reward = th.FloatTensor(batch['reward'])
        done = th.FloatTensor(batch['done'])
        old_action_log_prob = th.FloatTensor(batch['action_log_prob'])
        value = th.FloatTensor(batch['value'])

        # advantage




        J = th.zero((1,))
        TD_loss = th.zero((1,))

        for _ in range(self.ppo_epoch):
            minibatch_ind = np.random.choice(self.batch_size, self.minibatch_size, replace=False)
            minibatch_obs = obs[minibatch_ind]
            minibatch_action = action_index[minibatch_ind]
            minibatch_old_action_log_prob = old_action_log_prob[minibatch_ind]

            minibatch_value, minibatch_new_action_log_prob, minibatch_dist_entropy, _ \
                = self.ac.evaluate_actions(obs=minibatch_obs, action=action_index)

            ratio = th.exp(minibatch_old_action_log_prob - minibatch_new_action_log_prob)
            surr1 = ratio *






        Loss = J + TD_loss

        self.writer.add_scalar('Loss/J', J.item(), self._episode)
        self.writer.add_scalar('Loss/TD_loss', TD_loss.item(), self._episode)
        self.writer.add_scalar('Loss/loss', Loss.item(), self._episode)


        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, 10)
        self.optimiser.step()

        self._episode += 1




