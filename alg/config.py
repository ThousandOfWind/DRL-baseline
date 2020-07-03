PARAM = {}

REINFORCE = {
    'mamory_size': 10,
    'alg': 'REINFORCE',
    'soft': True,
    'gamma': 0.99,
    'learning_rate': 0.01,
    'n_episode': 10000,
    'hidden_dim': 64,
    'memory_type': 'ep'
}

QLearning = {
    'mamory_size': 100000,
    'alg': 'QLearning',
    'soft': False,
    'gamma': 0.99,
    'learning_rate': 0.01,
    'n_episode': 10000,
    'hidden_dim': 64,
    'batch_size': 64,
    'memory_type': 'os',
    'target_update_interval': 50000,
    'epsilon_start': 0.5,
    'epsilon_end': 0.0001,
    'time_length': 500000

}

PARAM['REINFORCE'] = REINFORCE
PARAM['QLearning'] = QLearning