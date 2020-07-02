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

        self.fc1 = nn.Linear(self.input_len, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_len)


    def forward(self, obs, **kwargs):
        x = th.tanh(self.fc1(obs))
        x = th.tanh(self.fc2(x))
        x = th.tanh(self.fc3(x))

        pi = F.softmax(th.exp(x))
        log_pi = th.log(pi)

        return pi, log_pi
