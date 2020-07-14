import torch.nn as nn
import torch.nn.functional as F
import torch as th



class DNNAgent(nn.Module):
    def __init__(self, param_set):
        super(DNNAgent, self).__init__()
        self.param_set = param_set

        self.input_len = param_set['obs_shape'][0]
        self.hidden_dim = param_set['hidden_dim'] # 64
        self.n_action = param_set['n_action']
        # self.dropout = nn.Dropout(p=0.6)

        self.soft = param_set['soft']
        self.use_rnn = param_set['rnn']

        self.fc1 = nn.Linear(self.input_len, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.n_action)

    def init_hidden(self):
        return th.zeros((1, self.hidden_dim))

    def forward(self, obs, hidden=None, **kwargs):
        x = F.relu(self.fc1(obs))

        if self.use_rnn:
            h_in = hidden.reshape(x.shape)
            h = F.relu(self.rnn(x, h_in))
            x = F.relu(self.fc2(h))
        else:
            x = F.relu(self.fc2(x))

        x = self.fc3(x)

        if self.soft:
            pi = F.softmax(th.exp(x), dim=-1)
            return (pi, h) if self.use_rnn else pi
        else:
            return (x, h) if self.use_rnn else x
