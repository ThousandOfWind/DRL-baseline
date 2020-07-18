import copy
import torch as th
from torch.optim import Adam
import torch.nn.functional as F

import numpy as np

import os
from .util.epsilon_schedules import DecayThenFlatSchedule

from .model.ddpg_base import DDPG_Critic, DDPG_Actor

class DDPG:
    """
    1. DQN- RNNAgent
    2. train
    """
    def __init__(self, param_set, writer):

        self.obs_shape = param_set['obs_shape'][0]
        self.gamma = param_set['gamma']
        self.learning_rate = param_set['learning_rate']
        self.n_action = param_set['n_action']

        self.Q = DDPG_Critic(param_set)
        self.actor = DDPG_Actor(param_set)

        self.targetQ = copy.deepcopy(self.Q)
        self.targetA = copy.deepcopy(self.actor)

        self.tau = param_set['tau']

        self.last_update = 0

        self.critic_optimiser = Adam(params=self.Q.parameters(), lr=self.learning_rate)
        self.actor_optimiser = Adam(params=self.actor.parameters(), lr=self.learning_rate)

        self.writer = writer
        self.step = 0
        self.batch_size = param_set['batch_size']

    def get_action(self, observation, greedy=False):
        obs = th.FloatTensor(observation)
        action = self.actor(obs)
        return action

    def learn(self, memory):
        batch = memory.get_sample(batch_size=self.batch_size)
        if not batch['flag']:
            return
        obs = th.FloatTensor(batch['observation'])
        action_index = th.LongTensor(batch['action_index'])
        next_obs = th.FloatTensor(batch['next_obs'])
        reward = th.FloatTensor(batch['reward'])
        done = th.FloatTensor(batch['done'])

        currentQ = self.Q(obs, action_index)
        targetQ = (reward + self.gamma * (1-done) * self.targetQ(next_obs, self.targetA(next_obs))).detach()
        critic_loss = F.mse_loss(currentQ, targetQ)
        self.writer.add_scalar('Loss/TD_loss', critic_loss.item(), self.step )


        # Optimize the critic
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()

        actor_loss = - self.Q(obs, self.actor(obs))
        self.writer.add_scalar('Loss/pi_loss', actor_loss.item(), self.step )

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        for param, target_param in zip(self.Q.parameters(), self.targetQ.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.targetA.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)




