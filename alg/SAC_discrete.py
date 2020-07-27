import copy
import torch as th
from torch.optim import Adam
import torch.nn.functional as F

import numpy as np

import os
from .util.epsilon_schedules import DecayThenFlatSchedule

from .model.sacd_base import SAC_discrete_Critic, SAC_discrete_Actor

class SAC_Discrete:
    def __init__(self, param_set, writer):

        self.obs_shape = param_set['obs_shape'][0]
        self.gamma = param_set['gamma']
        self.q_learning_rate = param_set['Q_learning_rate']
        self.policy_learning_rate = param_set['Policy_learning_rate']
        self.alpha_learning_rate = param_set['Alpha_learning_rate']
        self.learnable_alpha = param_set['learnable_alpha']
        self.soft_update = param_set['soft_update']

        self.n_action = param_set['n_action']

        self.Q = SAC_discrete_Critic(param_set)
        self.actor = SAC_discrete_Actor(param_set)
        self.targetQ = copy.deepcopy(self.Q)

        self.log_alpha = th.tensor(np.log(param_set['init_alpha']))
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -param_set['n_action']

        self.critic_optimiser = Adam(params=self.Q.parameters(), lr=self.q_learning_rate)
        self.actor_optimiser = Adam(params=self.actor.parameters(), lr=self.policy_learning_rate)
        if self.learnable_alpha:
            self.alpha_optimiser = Adam(params=[self.log_alpha], lr=self.alpha_learning_rate)

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
        action_index, action_log_probs, pi = self.actor(obs)
        return action_index, pi

    def learn(self, memory):
        batch = memory.get_sample(batch_size=self.batch_size)
        if not batch['flag']:
            return
        self.step += 1

        obs = th.FloatTensor(batch['observation'])
        action_index = th.LongTensor(batch['action_index']).unsqueeze(-1)
        next_obs = th.FloatTensor(batch['next_obs'])
        reward = th.FloatTensor(batch['reward']).unsqueeze(-1)
        done = th.FloatTensor(batch['done']).unsqueeze(-1)

        # targetnextQ1, targetnextQ2 = self.targetQ(next_obs)
        # next_log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True)
        # targetV = th.min(targetnextQ1, targetnextQ2) - self.alpha * next_log_prob

        currentQ1, currentQ2  = self.Q(obs)
        currentQ1 = currentQ1.gather(-1, action_index)
        currentQ2 = currentQ2.gather(-1, action_index)

        next_action_index, next_action_log_probs, next_pi = self.actor(next_obs)
        target_next_Q = th.min(*self.targetQ(next_obs))
        targetV = (next_pi * (target_next_Q - self.alpha * next_action_log_probs)).sum(dim=1, keepdim=True)

        targetQ = (reward + self.gamma * (1-done) * targetV).detach()
        critic_loss = F.mse_loss(currentQ1, targetQ) + F.mse_loss(currentQ2, targetQ)
        self.writer.add_scalar('Loss/TD_loss', critic_loss.item(), self.step )

        # Optimize the critic
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()


        if self.step % self.target_Q_update_frequncy == 0:
            if self.soft_update:
                for param, target_param in zip(self.Q.parameters(), self.targetQ.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            else:
                self.targetQ.load_state_dict(self.Q.state_dict())

        if self.step % self.pi_update_interval == 0:
            action_index, action_log_probs, pi = self.actor(obs)
            q1, q2 = self.Q(obs)
            q = (th.min(q1, q2) * pi).sum(dim=1, keepdim=True)
            entropies = -(action_log_probs * pi).sum(dim=1, keepdim=True)
            actor_loss = (- self.alpha.detach() * entropies - q).mean()

            self.writer.add_scalar('Loss/pi_loss', actor_loss.item(), self.step)
            self.writer.add_scalar('Loss/Entropy', entropies.mean().item(), self.step)

            self.actor_optimiser.zero_grad()
            actor_loss.backward()
            self.actor_optimiser.step()

            if self.learnable_alpha:
                self.alpha_optimiser.zero_grad()
                alpha_loss = (self.alpha *
                              (entropies.detach() - self.target_entropy).detach()).mean()
                self.writer.add_scalar('Loss/alpha_loss', alpha_loss.item(), self.step)

                self.writer.add_scalar('network/alpha', self.alpha.item(), self.step)

                alpha_loss.backward()
                self.alpha_optimiser.step()