import copy
import torch as th
from torch.optim import Adam
import torch.nn.functional as F

import numpy as np

import os
from .util.epsilon_schedules import DecayThenFlatSchedule

from .model.ddpg_base import TD3_Critic, SAC_Actor

class SAC:
    """
    1. DQN- RNNAgent
    2. train
    """
    def __init__(self, param_set, writer):

        self.obs_shape = param_set['obs_shape'][0]
        self.gamma = param_set['gamma']
        self.q_learning_rate = param_set['Q_learning_rate']
        self.policy_learning_rate = param_set['Policy_learning_rate']
        self.learnable_alpha = param_set['learnable_alpha']

        self.n_action = param_set['n_action']
        self.action_range = param_set['action_range']

        self.Q = TD3_Critic(param_set)
        self.actor = SAC_Actor(param_set)
        self.targetQ = copy.deepcopy(self.Q)

        self.log_alpha = th.tensor(np.log(param_set['init_alpha'])).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -param_set['n_action']

        self.critic_optimiser = Adam(params=self.Q.parameters(), lr=self.q_learning_rate)
        self.actor_optimiser = Adam(params=self.actor.parameters(), lr=self.policy_learning_rate)
        if self.learnable_alpha:
            self.alpha_optimiser = Adam(params=[self.log_alpha], lr=self.policy_learning_rate)

        self.tau = param_set['tau']
        self.step = 0
        self.target_Q_update_frequncy = param_set['target_Q_update_interval']
        self.pi_update_interval = param_set['pi_update_interval']

        self.writer = writer
        self.batch_size = param_set['batch_size']

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, observation, sample=False):
        obs = th.FloatTensor(observation)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        return action

    def learn(self, memory):
        batch = memory.get_sample(batch_size=self.batch_size)
        if not batch['flag']:
            return
        self.step += 1

        obs = th.FloatTensor(batch['observation'])
        action_index = th.LongTensor(batch['action_index'])
        next_obs = th.FloatTensor(batch['next_obs'])
        reward = th.FloatTensor(batch['reward'])
        done = th.FloatTensor(batch['done'])

        currentQ1, currentQ2  = self.Q(obs, action_index)

        next_dist = self.actor(next_obs)
        next_action = next_dist.rsample()
        targetnextQ1, targetnextQ2 = self.targetQ(next_obs, next_action)

        next_log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True)
        targetV = th.min(targetnextQ1, targetnextQ2) - self.alpha * next_log_prob
        targetQ = (reward + self.gamma * (1-done) * targetV).detach()
        critic_loss = F.mse_loss(currentQ1, targetQ) + F.mse_loss(currentQ2, targetQ)
        self.writer.add_scalar('Loss/TD_loss', critic_loss.item(), self.step )

        # Optimize the critic
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()


        if self.step % self.target_Q_update_frequncy == 0:
            for param, target_param in zip(self.Q.parameters(), self.targetQ.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.step % self.pi_update_interval == 0:
            dist = self.actor(obs)
            action = dist.rsample()
            q1, q2 = - self.Q(obs, action)
            q = th.min(q1, q2)

            # .sum(-1, keepdim=True) 感觉不应该存在呀
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)

            actor_loss = (self.alpha.detach() * log_prob - q).mean()

            self.writer.add_scalar('Loss/pi_loss', actor_loss.item(), self.step)
            self.actor_optimiser.zero_grad()
            actor_loss.backward()
            self.actor_optimiser.step()

            if self.learnable_alpha:
                self.alpha_optimiser.zero_grad()
                alpha_loss = (self.alpha *
                              (-log_prob - self.target_entropy).detach()).mean()
                self.writer.add_scalar('Loss/alpha_loss', alpha_loss.item(), self.step)
                self.writer.add_scalar('network/alpha', self.alpha.item(), self.step)

                alpha_loss.backward()
                self.alpha_optimiser.step()