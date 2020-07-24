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
        self.Q = DNNAgent(param_set)
        self.Q.soft = False
        self.Q.use_rnn = False

        self.params = [
                        {'params': self.pi.parameters()},
                        {'params': self.Q.parameters()}
                    ]
        self.optimiser = Adam(self.params, lr=self.learning_rate)
        self.writer = writer
        self._episode = 0

        self.log_pi_batch = []
        self.value_batch = []
        self.rnn = param_set['rnn']


    def new_trajectory(self):
        self.log_pi_batch = []
        self.value_batch = []
        self.hidden = self.pi.init_hidden()


    def get_action(self, observation, *arg):
        obs = th.FloatTensor(observation)
        if self.rnn:
            obs_ = obs.unsqueeze(0)
            pi, self.hidden = self.pi(obs=obs_, hidden=self.hidden)
            pi = pi.squeeze(0)
        else:
            pi = self.pi(obs=obs).squeeze(0)
        m = Categorical(pi)
        action_index = m.sample()

        self.log_pi_batch.append(m.log_prob(action_index))

        q = self.Q(obs=obs)
        self.value_batch.append(q[action_index])
        return int(action_index), pi

    def learn(self, memory):
        batch = memory.get_last_trajectory()

        reward = th.FloatTensor(batch['reward'][0])
        log_pi = th.stack(self.log_pi_batch)

        value = th.stack(self.value_batch)
        mask = th.ones_like(value)
        mask[-1] = 0
        next_value = th.cat([value[1:], value[0:1]],dim=-1) * mask

        td_error = reward + self.gamma * next_value.detach() - value
        td_loss = (td_error ** 2).mean()
        J = - (value.detach() * log_pi).mean()

        loss = J + td_loss

        self.writer.add_scalar('Loss/J', J.item(), self._episode)
        self.writer.add_scalar('Loss/TD_loss', td_loss.item(), self._episode)
        self.writer.add_scalar('Loss/loss', loss.item(), self._episode)


        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.pi.parameters(), 10)
        grad_norm = th.nn.utils.clip_grad_norm_(self.Q.parameters(), 10)

        self.optimiser.step()

        self._episode += 1




