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


class PPOLearner:
    """
    1. DQN- RNNAgent
    2. train
    """

    def __init__(self, param_set, writer):
        self.obs_shape = param_set['obs_shape'][0]
        self.gamma = param_set['gamma']
        self.lamda = param_set['lamda']
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
        self.lossvalue_norm = param_set['lossvalue_norm']
        self.loss_coeff_value = param_set['loss_coeff_value']
        self.loss_coeff_entropy = param_set['loss_coeff_entropy']


    def get_action(self, observation, *arg):
        obs = th.FloatTensor(observation)
        action_index, value, action_log_probs, _ = self.ac.select_action(obs=obs)
        return int(action_index), action_log_probs.detach().numpy(), value.detach().numpy()

    def learn(self, memory):
        batch = memory.get_current_trajectory()
        if len(batch['observation']) < self.batch_size:
            return

        obs = th.FloatTensor(batch['observation'])
        action_index = th.LongTensor(batch['action_index'])
        next_obs = th.FloatTensor(batch['next_obs'])
        reward = th.FloatTensor(batch['reward'])
        done = th.FloatTensor(batch['done'])
        batch['action_log_prob'] = np.stack(batch['action_log_prob'], axis=0)
        old_action_log_prob = th.FloatTensor(batch['action_log_prob'])
        value = th.FloatTensor(batch['value'])

        # advantage
        advangtage = th.zeros_like(reward)
        returns = th.zeros_like(reward)
        deltas = th.zeros_like(reward)
        pre_return = 0
        pre_value = 0
        pre_advantage = 0
        for i in range(advangtage.shape[0]-1, -1, -1):
            returns[i] = reward[i] + (1 - done[i]) * self.gamma * pre_return
            deltas[i] = reward[i] + (1 - done[i]) * self.gamma * pre_value - value[i]
            advangtage[i] = deltas[i] + (1 - done[i]) * self.gamma * self.lamda * pre_advantage
            pre_return = returns[i]
            pre_value = value[i]
            pre_advantage = advangtage[i]

        # # also
        # pre_return = 0
        # for i in range(advangtage.shape[0]-1, -1, -1):
        #     pre_return = reward[i] + (1 - done[i]) * self.gamma * pre_return
        #     advangtage[i] = pre_return - value[i]

        L_V = []
        L_P = []
        L_E = []

        for _ in range(self.ppo_epoch):
            minibatch_ind = np.random.choice(self.batch_size, self.minibatch_size, replace=False)
            minibatch_obs = obs[minibatch_ind]
            minibatch_old_action_log_prob = old_action_log_prob[minibatch_ind]
            minibatch_advantange = advangtage[minibatch_ind]
            minibatch_return = returns[minibatch_ind]
            minibatch_action_index = action_index[minibatch_ind]


            minibatch_new_value, minibatch_new_action_log_prob, minibatch_dist_entropy, _ \
                = self.ac.evaluate_actions(obs=minibatch_obs, action=minibatch_action_index)
            ratio = th.exp(minibatch_old_action_log_prob - minibatch_new_action_log_prob)
            surr1 = ratio * minibatch_advantange
            surr2 = ratio.clamp( 1 - self.clip, 1 + self.clip)
            loss_surr = - th.mean(th.min(surr1, surr2))

            if self.lossvalue_norm:
                minibatch_return_6std = 6 * minibatch_return.std()
                loss_value = th.mean((minibatch_new_value - minibatch_return).pow(2))/minibatch_return_6std
            else:
                loss_value = th.mean((minibatch_new_value - minibatch_return).pow(2))

            loss_entropy = th.mean(th.exp(minibatch_new_action_log_prob) * minibatch_new_action_log_prob)

            total_loss = loss_surr + self.loss_coeff_value * loss_value + self.loss_coeff_entropy * loss_entropy

            self.optimiser.zero_grad()
            total_loss.backward()
            self.optimiser.step()

            L_P.append(loss_surr.item())
            L_V.append(loss_value.item())
            L_E.append(loss_entropy.item())



        self.writer.add_scalar('Loss/J', sum(L_P), self._episode)
        self.writer.add_scalar('Loss/B', sum(L_V), self._episode)
        self.writer.add_scalar('Loss/Entropy', sum(L_E), self._episode)

        self.writer.add_scalar('Loss/loss', sum(L_P) + sum(L_V) + sum(L_E), self._episode)



        self._episode += 1




