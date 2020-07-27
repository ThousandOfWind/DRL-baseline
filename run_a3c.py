import gym
import random
import cv2 as cv
import numpy as np
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp


from alg import Learner
from mem import Memory
from alg.config import PARAM
from alg.model import ac_sharenet
from alg.util.my_adam import SharedAdam

import copy
import numpy as np
import torch as th

ENV_NAME = 'Asterix-ram-v0' #'CartPole-v1'  # Environment name

EPISODIC_TRAIN = ['REINFORCE', 'REINFORCE-B', 'QAC', 'QAC-SN', 'OS-AC', 'QAC-rnn']


def init():
    env = gym.make(ENV_NAME)
    param_set = copy.deepcopy(PARAM['A3C'])
    env_param = {
        'test_episode': 4,
        'random_seed': random.randint(0, 1000),
        'n_action': env.action_space.n,
        'obs_shape': env.observation_space.shape,
        'max_step': env._max_episode_steps,
    }
    param_set.update(env_param)

    np.random.seed(param_set['random_seed'])
    # th.manual_seed(param_set['random_seed'])

    param_set['path'] = ENV_NAME + '/' + param_set['alg'] + '/' + str(param_set['random_seed'])

    writer = SummaryWriter('logs/' + param_set['path'])
    env.close()

    return writer, param_set

def run(writer, param_set, share_model):
    param_set['worker_id'] = i
    agent = Learner['A3C'](param_set, writer, share_model, optimizer)
    memory = Memory[param_set['memory_type']](param_set)
    env = gym.make(ENV_NAME)

    for e in range(param_set['n_episode']):
        terminal = False
        observation = env.reset()
        step = 0
        total_reward = 0

        if param_set['alg'] in EPISODIC_TRAIN:
            agent.new_trajectory()

        while not terminal and step < param_set['max_step']:
            action, _ = agent.get_action(observation)

            next_observation, reward, terminal, _ = env.step(action=action)
            # env.render()

            if param_set['memory_type'] == 'ep':

                experience = {
                    'observation': observation,
                    'action_index': action,
                    'reward': reward,
                }
                memory.append(experience)
            else:
                experience = {
                    'observation': observation,
                    'action_index': action,
                    'reward': reward,
                    'next_obs': next_observation,
                    'done': 1 if terminal else 0
                }
                memory.append(experience)
                agent.learn(memory)


            observation = next_observation
            step += 1
            total_reward += reward

        if param_set['memory_type'] == 'ep':
            experience = {
                'observation': observation,
            }
            memory.end_trajectory(experience)
            agent.learn(memory)

        print(e,':',total_reward, '/', step)
        writer.add_scalar('data/reward_avg_step', total_reward / step, e)
        writer.add_scalar('data/reward', total_reward, e)

class Util:
    def __init__(self, high, low, size):
        self.I_h = high[0]
        self.J_h = high[2]
        self.I_l = low[0]
        self.J_l = low[2]

        self.I_scale = size / (self.I_h - self.I_l)
        self.J_scale = size / (self.J_h - self.J_l)

    def get_pos(self, obs):
        i = obs[0]
        j = obs[2]
        i = int((i-self.I_l)* self.I_scale)
        j = int((j-self.J_l)* self.J_scale)
        return (i,j)


def test(env, agent, param_set):
    # size = 1000
    # img_action_intend = np.zeros([size, size, 3], np.float32)
    # util = Util(env.observation_space.high, env.observation_space.low, size)

    for e in range(param_set['test_episode']):
        terminal = False
        observation = env.reset()
        step = 0
        total_reward = 0
        if param_set['alg'] in EPISODIC_TRAIN:
            agent.new_trajectory()

        while not terminal and step < param_set['max_step']:
            action, prop = agent.get_action(observation, True)

            next_observation, reward, terminal, _ = env.step(action=action)
            # pos = util.get_pos(observation)
            # img_action_intend[pos[0],pos[1], 1:] += prop.detach().numpy()
            observation = next_observation
            step += 1
            total_reward += reward

        print('test:',total_reward, '/', step)

    # min = img_action_intend.min()
    # print(min)
    # img_action_intend[img_action_intend==0] = min
    # img_action_intend = img_action_intend - min
    # img_action_intend = (img_action_intend/ img_action_intend.max()) * 255
    # img_action_intend.astype(np.uint8)
    #
    # cv.imshow("img_action_intend", img_action_intend)
    # cv.waitKey(0)



if __name__ == '__main__':
    writer, param_set = init()

    share_model = ac_sharenet(param_set)
    optimizer = SharedAdam(share_model.parameters(), lr=param_set['learning_rate'])
    optimizer.share_memory()
    processes = []

    for i in range(param_set['num_processes']):
        p = mp.Process(target=run, args=(writer, param_set, share_model, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
