import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import Categorical
from ..util.distribution import TanhNormal



class DDPG_Critic( nn.Module):
    def __init__(self, param_set):
        super(DDPG_Critic, self).__init__()
        input_len = param_set['obs_shape'][0] + param_set['n_action']
        self.hidden_dim = param_set['hidden_dim']
        layer_norm = param_set['layer_norm']

        self.critic_fc1 = nn.Linear(input_len, self.hidden_dim)
        self.critic_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.critic_fc3 = nn.Linear(self.hidden_dim, 1)

        if layer_norm:
            self.layer_norm(self.critic_fc1)
            self.layer_norm(self.critic_fc2)
            self.layer_norm(self.critic_fc3)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        th.nn.init.orthogonal_(layer.weight, std)
        th.nn.init.constant_(layer.bias, bias_const)

    def forward(self, obs, action):
        x = th.cat([obs, action], dim=-1)
        x = F.tanh(self.critic_fc1(x))
        x = F.tanh(self.critic_fc2(x))
        critic = self.critic_fc3(x)
        return critic

class DDPG_Actor( nn.Module):
    def __init__(self, param_set):
        super(DDPG_Actor, self).__init__()
        input_len = param_set['obs_shape'][0]
        self.hidden_dim = param_set['hidden_dim']
        self.continue_action = param_set['continue']
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
        action = self.actor_fc3(x)
        return action

class TD3_Critic( nn.Module):
    def __init__(self, param_set):
        super(DDPG_Critic, self).__init__()
        input_len = param_set['obs_shape'][0] + param_set['n_action']
        self.hidden_dim = param_set['hidden_dim']
        layer_norm = param_set['layer_norm']

        self.critic_fc1 = nn.Linear(input_len, self.hidden_dim)
        self.critic_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.critic_fc3 = nn.Linear(self.hidden_dim, 1)

        self.critic_fc4 = nn.Linear(input_len, self.hidden_dim)
        self.critic_fc5 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.critic_fc6 = nn.Linear(self.hidden_dim, 1)

        if layer_norm:
            self.layer_norm(self.critic_fc1)
            self.layer_norm(self.critic_fc2)
            self.layer_norm(self.critic_fc3)

            self.layer_norm(self.critic_fc4)
            self.layer_norm(self.critic_fc5)
            self.layer_norm(self.critic_fc6)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        th.nn.init.orthogonal_(layer.weight, std)
        th.nn.init.constant_(layer.bias, bias_const)

    def forward(self, obs, action):
        x = th.cat([obs, action], dim=-1)
        x = F.tanh(self.critic_fc1(x))
        x = F.tanh(self.critic_fc2(x))
        critic = self.critic_fc3(x)

        x2 = th.cat([obs, action], dim=-1)
        x2 = F.tanh(self.critic_fc4(x2))
        x2 = F.tanh(self.critic_fc5(x2))
        critic2 = self.critic_fc6(x2)
        return critic, critic2

class SAC_Actor( nn.Module):
    def __init__(self, param_set):
        super(SAC_Actor, self).__init__()
        input_len = param_set['obs_shape'][0]
        self.hidden_dim = param_set['hidden_dim']
        self.continue_action = param_set['continue']
        layer_norm = param_set['layer_norm']
        self.log_std_bound = param_set['log_std_bound']

        self.actor_fc1 = nn.Linear(input_len, self.hidden_dim)
        self.actor_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.actor_fc3 = nn.Linear(self.hidden_dim, 2 * param_set['n_action'])

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
        mu, log_std = self.actor_fc3(x).chunk(2, dim=-1)

        log_std = th.clamp(log_std, self.log_std_bound[0], self.log_std_bound[1])
        std = log_std.exp()

        return TanhNormal(mu, std)