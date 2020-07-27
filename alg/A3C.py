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

    def __init__(self, param_set, writer, share_model:DNNAgent):
        self.obs_shape = param_set['obs_shape'][0]
        self.gamma = param_set['gamma']
        self.learning_rate = param_set['learning_rate']
        self.clone_share_model = param_set['clone_share_model']
        self.id = param_set['worker_id']


        if self.clone_share_model:
            self.ac = copy.deepcopy(share_model)
            self.soft_clone = param_set['soft_clone']
            if self.soft_clone:
                self.tau = param_set['tau']
        else:
            self.ac = DNNAgent(param_set)

        self.params = share_model.parameters()
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

    def ensure_shared_grads(self):
        for param, shared_param in zip(self.ac.parameters(),
                                       self.params):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def clone_to_local(self):
        if self.soft_clone:
            for param, target_param in zip(self.ac.parameters(), self.params):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            for param, target_param in zip(self.ac.parameters(), self.params):
                target_param.data.copy_(target_param.data)

    def learn(self, memory):
        batch = memory.get_last_trajectory()

        reward = th.FloatTensor(batch['reward'][0])
        value = th.stack(self.value_batch)
        log_pi = th.stack(self.log_pi_batch)


        # advantage
        advangtage = th.zeros_like(reward)
        returns = th.zeros_like(reward)
        deltas = th.zeros_like(reward)
        pre_return = 0
        pre_value = 0
        pre_advantage = 0
        for i in range(advangtage.shape[0]-1, -1, -1):
            returns[i] = reward[i] + self.gamma * pre_return
            deltas[i] = reward[i] + self.gamma * pre_value - value[i]
            advangtage[i] = deltas[i] + self.gamma * self.lamda * pre_advantage
            pre_return = returns[i]
            pre_value = value[i]
            pre_advantage = advangtage[i]


        mask = th.ones_like(value)
        mask[-1] = 0
        next_value = th.cat([value[1:], value[0:1]],dim=-1) * mask

        td_error = reward + self.gamma * next_value.detach() - value
        td_loss = (td_error ** 2).mean()
        J = - (advangtage.detach() * log_pi).mean()

        loss = J + td_loss

        self.writer.add_scalar('Loss/J_' + str(self.id), J.item(), self._episode)
        self.writer.add_scalar('Loss/TD_loss_' + str(self.id), td_loss.item(), self._episode)
        self.writer.add_scalar('Loss/loss_' + str(self.id), loss.item(), self._episode)


        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, 10)
        self.ensure_shared_grads()
        self.optimiser.step()

        if self.clone_share_model:
            self.clone_to_local()

        self._episode += 1




