import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import Categorical


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.fc = nn.Linear(recurrent_input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        x = F.relu(self.fc(x))
        x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
        x = x.squeeze(0)
        hxs = hxs.squeeze(0)
        return x, hxs


class MLPBase(NNBase):
    def __init__(self, param_set):

        input_len = param_set['obs_shape'][0]
        self.hidden_dim = param_set['hidden_dim']
        n_action = param_set['n_action']
        recurrent = param_set['rnn']
        layer_norm = param_set['layer_norm']
        self.min_pi = param_set['min_pi']

        super(MLPBase, self).__init__(recurrent, input_len, self.hidden_dim)

        if recurrent:
            input_len = self.hidden_dim


        self.actor_fc1 = nn.Linear(input_len, self.hidden_dim)
        self.actor_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.actor_fc3 = nn.Linear(self.hidden_dim, n_action)

        self.critic_fc1 = nn.Linear(input_len, self.hidden_dim)
        self.critic_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.critic_fc3 = nn.Linear(self.hidden_dim, 1)

        if layer_norm:
            self.layer_norm(self.actor_fc1)
            self.layer_norm(self.actor_fc2)
            self.layer_norm(self.actor_fc3)

            self.layer_norm(self.critic_fc1)
            self.layer_norm(self.critic_fc2)
            self.layer_norm(self.critic_fc3)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        th.nn.init.orthogonal_(layer.weight, std)
        th.nn.init.constant_(layer.bias, bias_const)

    def init_hidden(self):
        return th.zeros((1, self.hidden_dim))

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        a = F.tanh(self.actor_fc1(x))
        a = F.tanh(self.actor_fc2(a))
        a = th.exp(self.actor_fc3(a))
        a = th.clamp(a, min=self.min_pi)
        action = F.softmax(a, dim=-1)

        c = F.tanh(self.critic_fc1(x))
        c = F.tanh(self.critic_fc2(c))
        critic = self.critic_fc3(c)



        return action, critic, rnn_hxs

    def forward_critic(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = F.tanh(self.critic_fc1(x))
        x = F.tanh(self.critic_fc2(x))
        critic = self.critic_fc3(x)
        return critic, rnn_hxs


class Policy(nn.Module):
    def __init__(self, param_set):
        super(Policy, self).__init__()
        self.base = MLPBase(param_set)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs=None, masks=None):
        raise NotImplementedError

    def select_action(self, obs, rnn_hxs=None, masks=None, deterministic=False):
        pi, value, rnn_hxs = self.base.forward(obs, rnn_hxs, masks)

        m = Categorical(pi)
        action_index = m.sample()
        action_log_probs = m.log_prob(action_index)

        return action_index, value, action_log_probs, rnn_hxs

    def get_value(self, obs, rnn_hxs=None, masks=None):
        value, rnn_hxs = self.base.forward_critic(obs, rnn_hxs, masks)
        return value, rnn_hxs

    def evaluate_actions(self, obs, action, last_rnn_hxs=None, masks=None):
        pi, value, rnn_hxs = self.base.forward(obs, last_rnn_hxs, masks)

        m = Categorical(pi)
        action_log_probs = m.log_prob(action)
        dist_entropy = m.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

