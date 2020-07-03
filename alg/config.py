PARAM = {}

REINFORCE = {
    'mamory_size': 10,
    'alg': 'REINFORCE',
    'soft': True,
    'gamma': 0.9,
    'learning_rate': 0.01,
    'n_episode': 200,
    'hidden_dim': 32,
    'memory_type': 'ep'
}

QLearning = {
    'mamory_size': 1000000,
    'alg': 'QLearning',
    'soft': False,
    'gamma': 0.9,
    'learning_rate': 0.01,
    'n_episode': 200,
    'hidden_dim': 32,
    'memory_type': 'os',
    'target_update_interval': 200,
    'epsilon': 0.1
}

PARAM['REINFORCE'] = REINFORCE
PARAM['QLearning'] = QLearning