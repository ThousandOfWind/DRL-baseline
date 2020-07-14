import torch as th
import random as rd
import copy

class Memmory:
    def __init__(self, param_set):
        self.trajectories = []
        self.max_trajextory_len = param_set['mamory_size']
        self.n_trajextories = 0

        self.current_trajectory = {
            'observation': [],
            'action_index': [],
            'next_obs': [],
            'reward': [],
            'done': []
        }

    def append(self, exprience):
        for key in exprience:
            if key in self.current_trajectory:
                self.current_trajectory[key].append(exprience[key])
                if self.current_trajectory[key].__len__() > self.max_trajextory_len:
                    self.current_trajectory[key] = self.current_trajectory[key][1:]

    def get_e(self, batch, e):
        batch['observation'].append(copy.deepcopy(self.current_trajectory['observation'][e]))
        batch['next_obs'].append(copy.deepcopy(self.current_trajectory['observation'][e]))
        batch['action_index'].append(copy.deepcopy(self.current_trajectory['action_index'][e]))
        batch['reward'].append(copy.deepcopy(self.current_trajectory['reward'][e]))
        batch['done'].append(copy.deepcopy(self.current_trajectory['done'][e]))


    def get_sample(self, batch_size=32, mode='random'):
        """
        目前来看 seq_len 都是max，但后面会不会有不同 如果不足就需要补充
        :param batch_size:
        :return:
        """
        if self.current_trajectory['done'].__len__() < batch_size:
            return {'flag': False}

        batch = {
            'flag': True,
            'observation': [],
            'next_obs': [],
            'action_index': [],
            'action_log_prob':[],
            'reward': [],
            'done': [],
            'value':[],
        }

        trajectory_len = self.current_trajectory['done'].__len__()

        for i in range(1, batch_size+1):
            e = rd.randint(0, trajectory_len-2) if mode == 'random' else -i
            self.get_e(batch, e)

        return batch


    def get_current_trajectory(self):
        batch = copy.deepcopy(self.current_trajectory)
        return batch

    def get_last_trajectory(self):
        batch = {
            'observation': [],
            'next_obs': [],
            'action_index': [],
            'reward': [],
            'done': [],
        }
        self.get_e(batch, -1)
        return batch

