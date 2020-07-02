import torch as th
import random as rd
import copy

class Memmory:
    """
    unimplement!!!

    """

    def __init__(self, param_set):
        self.trajectories = []
        self.max_n_trajextories = param_set['mamory_size']
        self.episode_index = 0
        self.n_trajextories = 0

        self.current_trajectory = {
            'observation': [],
            'action_index': [],
            'reward': [],
        }

    def append(self, exprience):
        for key in exprience:
            if key in self.current_trajectory:
                self.current_trajectory[key].append(exprience[key])

    def end_trajectory(self, exprience):
        self.append(exprience)
        self.n_trajextories += 1

        self.trajectories.append(copy.deepcopy(self.current_trajectory))

        self.episode_index += 1
        self.current_trajectory.clear()

        self.current_trajectory = {
            'observation': [],
            'action_index': [],
            'reward': [],
        }

        if self.trajectories.__len__() > self.max_n_trajextories:
            # self.trajectories.popleft()

            e = rd.randint(0, int(self.max_n_trajextories/2))
            self.trajectories.pop(e)

    def get_e(self,batch, e):
        traject = self.trajectories[e]
        batch['observation'].append(copy.deepcopy(traject['observation'][:-1]))
        batch['next_obs'].append(copy.deepcopy(traject['observation'][1:]))
        batch['action_index'].append(copy.deepcopy(traject['action_index']))
        batch['reward'].append(copy.deepcopy(traject['reward']))
        done = [0] * len(traject['observation'])
        done[-1] = 1
        batch['done'].append(done)

    def get_sample(self, batch_size=32, mode='random'):
        """
        目前来看 seq_len 都是max，但后面会不会有不同 如果不足就需要补充
        :param batch_size:
        :return:
        """
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

        batch = {
            'observation': [],
            'next_obs': [],
            'action_index': [],
            'reward': [],
            'done': [],
        }

        trajectory_len = self.trajectories.__len__()

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
        batch = copy.deepcopy(self.get_e(batch, -1))
        return batch
