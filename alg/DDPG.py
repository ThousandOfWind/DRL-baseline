import copy
import torch as th
from torch.optim import Adam
import numpy as np

import os
from .model.critic import DNN
from .util.epsilon_schedules import DecayThenFlatSchedule


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

        self.Q = DNN(param_set)
        self.targetQ = copy.deepcopy(self.Q)
        self.update_frequncy = param_set['target_update_interval']
        self.last_update = 0

        self.params = self.Q.parameters()
        self.optimiser = Adam(params=self.params, lr=self.learning_rate)
        self.writer = writer
        self.step = 0
        self.batch_size = param_set['batch_size']

        self.schedule = DecayThenFlatSchedule(start=param_set['epsilon_start'], finish=param_set['epsilon_end'],
                                              time_length=param_set['time_length'], decay="linear")

    def get_action(self, observation, greedy=False):
        obs = th.FloatTensor(observation)
        if np.random.rand() < self.schedule.eval(self.step) and not greedy:
            action_index = np.random.randint(0, self.n_action)
            q = th.full((self.n_action,), 1/self.n_action)
        else:
            q = self.Q(obs=obs)
            # print(q)
            action_index = int(q.argmax())

        return action_index, q

    def learn(self, memory):
        batch = memory.get_sample(batch_size=self.batch_size)
        if not batch['flag']:
            return
        obs = th.FloatTensor(batch['observation'])
        action_index = th.LongTensor(batch['action_index'])
        next_obs = th.FloatTensor(batch['next_obs'])
        reward = th.FloatTensor(batch['reward'])
        done = th.FloatTensor(batch['done'])

        q = self.Q(obs)
        chose_q = th.gather(q, dim=1, index=action_index.unsqueeze(-1)).squeeze(-1)

        next_qmax = self.targetQ(next_obs).max().squeeze(-1)
        target_q = reward + self.gamma * (1-done) * next_qmax
        td_error = chose_q - target_q.detach()
        td_loss = (td_error ** 2).mean()

        self.writer.add_scalar('Loss/TD_loss', td_loss.item(), self.step )
        self.optimiser.zero_grad()
        td_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, 10)
        self.optimiser.step()

        self.step += 1
        if self.step - self.last_update > self.update_frequncy:
            self.last_update = self.step
            self.targetQ.load_state_dict(self.Q.state_dict())



