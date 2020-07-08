PARAM = {}

REINFORCE = {
    'mamory_size': 10,
    'alg': 'REINFORCE',
    'soft': True,
    'gamma': 0.99,
    'learning_rate': 0.01,
    'n_episode': 5000,
    'hidden_dim': 128,
    'memory_type': 'ep'
}

REINFORCE_B = {
    'mamory_size': 10,
    'alg': 'REINFORCE-B',
    'soft': True,
    'gamma': 0.99,
    'learning_rate': 0.01,
    'n_episode': 5000,
    'hidden_dim': 128,
    'memory_type': 'ep'
}

QLearning = {
    'mamory_size': 100000,
    'alg': 'QLearning',
    'soft': False,
    'gamma': 0.99,
    'learning_rate': 0.001,
    'n_episode': 5000,
    'hidden_dim': 64,
    'batch_size': 128,
    'memory_type': 'os',
    'target_update_interval': 40000,
    'epsilon_start': 0.5,
    'epsilon_end': 0.01,
    'time_length': 40000

}

QAC = {
    'mamory_size': 10,
    'alg': 'QAC',
    'soft': True,
    'gamma': 0.99,
    'learning_rate': 0.01,
    'n_episode': 5000,
    'hidden_dim': 128,
    'memory_type': 'ep'
}

QAC_SN = {
    'mamory_size': 10,
    'alg': 'QAC-SN',
    'gamma': 0.99,
    'learning_rate': 0.01,
    'n_episode': 5000,
    'hidden_dim': 128,
    'memory_type': 'ep'
}


OS_AC = {
    'mamory_size': 10,
    'alg': 'OS-AC',
    'soft': True,
    'gamma': 0.99,
    'learning_rate': 0.01,
    'n_episode': 5000,
    'hidden_dim': 128,
    'memory_type': 'os'
}

PARAM['REINFORCE'] = REINFORCE
PARAM['QLearning'] = QLearning
PARAM['REINFORCE-B'] = REINFORCE_B
PARAM['QAC'] = QAC
PARAM['QAC-SN'] = QAC_SN
PARAM['OS-AC'] = OS_AC