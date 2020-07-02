import gym
import random

from alg import Learner
from mem.memory import Memmory


ENV_NAME = 'MountainCar-v0'  # Environment name
from tensorboardX import SummaryWriter


def init():
    env = gym.make(ENV_NAME)

    param_set = {
        'mamory_size': 10,
        'alg': 'REINFORCE',
        'gamma': 0.9,
        'learning_rate': 0.01,
        'random_seed': random.randint(0, 1000),
        'n_action': env.action_space.n,
        'obs_shape': env.observation_space.shape,
        'max_step': env._max_episode_steps,
        'n_episode': 100,
        'hidden_dim': 64,
    }

    param_set['path'] = ENV_NAME + '/' + param_set['alg'] + '/' + str(param_set['random_seed'])

    writer = SummaryWriter('logs/' + param_set['path'])
    agent = Learner[param_set['alg']](param_set, writer)

    memory = Memmory(param_set)

    run(env, agent, memory, writer, param_set)


def run(env, agent, memory, writer, param_set):
    for e in range(param_set['n_episode']):
        terminal = False
        observation = env.reset()
        step = 0
        total_reward = 0
        while not terminal and step < param_set['max_step']:
            action = agent.get_action(observation)
            next_observation, reward, terminal, _ = env.step(action)

            experience = {
            'observation': observation,
            'action_index': action,
            'reward': reward,
            }
            memory.append(experience)

            observation = next_observation
            step += 1
            total_reward += reward

        experience = {
            'observation': observation,
        }
        memory.append(experience)
        writer.add_scalar('data/reward_avg_step', total_reward / step, e)
        writer.add_scalar('data/reward', total_reward, e)
        agent.learn(memory)

if __name__ == '__main__':
    init()