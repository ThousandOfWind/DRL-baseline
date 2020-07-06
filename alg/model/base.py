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

        self.fc1 = nn.Linear(self.input_len, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.n_action)


    def forward(self, obs, **kwargs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)

        if self.soft:
            pi = F.softmax(th.exp(x), dim=-1)
            # print(pi)
            return pi
        else:
            return x
