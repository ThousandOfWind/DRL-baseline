import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import Categorical


class DNN(nn.Module):
    def __init__(self, input_len, hidden_dim, out_dim, layer_norm, av=False):
        super(DNN, self).__init__()

        self.av = av

        if self.av:
            self.a_head_fc1 = nn.Linear(input_len, hidden_dim)
            self.a_head_fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.a_head_fc3 = nn.Linear(hidden_dim, out_dim)

            self.v_head_fc1 = nn.Linear(input_len, hidden_dim)
            self.v_head_fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.v_head_fc3 = nn.Linear(hidden_dim, 1)

            if layer_norm:
                self.layer_norm(self.a_head_fc1)
                self.layer_norm(self.a_head_fc2)
                self.layer_norm(self.a_head_fc3)
                self.layer_norm(self.v_head_fc1)
                self.layer_norm(self.v_head_fc2)
                self.layer_norm(self.v_head_fc3)
        else:
            self.critic_fc1 = nn.Linear(input_len, hidden_dim)
            self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.critic_fc3 = nn.Linear(hidden_dim, out_dim)

            if layer_norm:
                self.layer_norm(self.critic_fc1)
                self.layer_norm(self.critic_fc2)
                self.layer_norm(self.critic_fc3)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        th.nn.init.orthogonal_(layer.weight, std)
        th.nn.init.constant_(layer.bias, bias_const)

    def forward(self, obs):
        if self.av:
            a = F.tanh(self.a_head_fc1(obs))
            a = F.tanh(self.a_head_fc2(a))
            a = self.a_head_fc3(a)

            v = F.tanh(self.v_head_fc1(obs))
            v = F.tanh(self.v_head_fc2(v))
            v = self.v_head_fc3(v)

            return v + a -a.mean(-1, keepdim=True)
        else:
            x = F.tanh(self.critic_fc1(obs))
            x = F.tanh(self.critic_fc2(x))
            critic = self.critic_fc3(x)

            return critic


class SAC_discrete_Critic( nn.Module):
    def __init__(self, param_set):
        super(SAC_discrete_Critic, self).__init__()
        input_len = param_set['obs_shape'][0]
        n_action = param_set['n_action']
        self.hidden_dim = param_set['hidden_dim']
        layer_norm = param_set['layer_norm']
        av = param_set['apart_av']

        self.critic1 = DNN(input_len, self.hidden_dim, n_action, layer_norm, av)
        self.critic2 = DNN(input_len, self.hidden_dim, n_action, layer_norm, av)


    def forward(self, obs):
        critic1 = self.critic1(obs)
        critic2 = self.critic2(obs)
        return critic1, critic2

class SAC_discrete_Actor( nn.Module):
    def __init__(self, param_set):
        super(SAC_discrete_Actor, self).__init__()
        input_len = param_set['obs_shape'][0]
        self.hidden_dim = param_set['hidden_dim']
        layer_norm = param_set['layer_norm']

        self.actor_fc1 = nn.Linear(input_len, self.hidden_dim)
        self.actor_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.actor_fc3 = nn.Linear(self.hidden_dim, param_set['n_action'])

        if layer_norm:
            self.layer_norm(self.actor_fc1)
            self.layer_norm(self.actor_fc2)
            self.layer_norm(self.actor_fc3)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        th.nn.init.orthogonal_(layer.weight, std)
        th.nn.init.constant_(layer.bias, bias_const)

    def forward(self, obs):
        x = obs
        x = F.tanh(self.actor_fc1(x))
        x = F.tanh(self.actor_fc2(x))
        x = self.actor_fc3(x)

        # x = th.clamp(x, min=self.min_pi)
        pi = F.softmax(x, dim=-1)
        m = Categorical(pi)
        action_index = m.sample()

        z = (pi == 0.0).float() * 1e-8
        action_log_probs = (pi+z).log()

        return action_index, action_log_probs, pi