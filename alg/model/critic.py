import torch.nn as nn
import torch.nn.functional as F
import torch as th



class DNN(nn.Module):
    def __init__(self, param_set):
        super(DNN, self).__init__()
        self.param_set = param_set

        self.input_len = param_set['obs_shape'][0]
        self.hidden_dim = param_set['hidden_dim'] # 64

        self.fc1 = nn.Linear(self.input_len, self.hidden_dim)
        # self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 1)


    def forward(self, obs, **kwargs):
        x = F.relu(self.fc1(obs))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
