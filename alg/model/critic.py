import torch.nn as nn
import torch.nn.functional as F
import torch as th



class DNN(nn.Module):
    def __init__(self, param_set, action_dim=None):
        super(DNN, self).__init__()
        self.param_set = param_set

        self.input_len = param_set['obs_shape'][0]
        self.hidden_dim = param_set['hidden_dim'] # 64
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.input_len, self.hidden_dim)
        if action_dim is None:
            self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.fc2 = nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 1)


    def forward(self, obs, action=None, **kwargs):
        x = F.relu(self.fc1(obs))
        if not self.action_dim is None:
            x = th.cat([x, action], dim=-1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
