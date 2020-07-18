PARAM = {}

REINFORCE = {
    'mamory_size': 10,
    'alg': 'REINFORCE',
    'soft': True,
    'gamma': 0.99,
    'learning_rate': 0.01,
    'n_episode': 5000,
    'hidden_dim': 128,
    'memory_type': 'ep',
    'rnn': False,
}

REINFORCE_B = {
    'mamory_size': 10,
    'alg': 'REINFORCE-B',
    'soft': True,
    'gamma': 0.99,
    'learning_rate': 0.01,
    'n_episode': 5000,
    'hidden_dim': 128,
    'memory_type': 'ep',
    'rnn': False,
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
    'time_length': 40000,
    'rnn': False,
}

QAC = {
    'mamory_size': 10,
    'alg': 'QAC',
    'soft': True,
    'gamma': 0.99,
    'learning_rate': 0.01,
    'n_episode': 5000,
    'hidden_dim': 128,
    'memory_type': 'ep',
    'rnn': False,
}

QAC_SN = {
    'mamory_size': 10,
    'alg': 'QAC-SN',
    'gamma': 0.99,
    'learning_rate': 0.01,
    'n_episode': 5000,
    'hidden_dim': 128,
    'memory_type': 'ep',
    'rnn': False,
}


OS_AC = {
    'mamory_size': 10,
    'alg': 'OS-AC',
    'soft': True,
    'gamma': 0.99,
    'pi_learning_rate': 0.005,
    'V_learning_rate': 0.1,
    'n_episode': 5000,
    'hidden_dim': 128,
    'memory_type': 'os',
    'rnn': False,
}
QAC_rnn = {
    'mamory_size': 10,
    'alg': 'QAC',
    'soft': True,
    'gamma': 0.99,
    'learning_rate': 0.01,
    'n_episode': 5000,
    'hidden_dim': 128,
    'memory_type': 'ep',
    'rnn': True,
}

PPO2 = {
    'mamory_size': 1000,
    'alg': 'PPO2',
    'soft': True,
    'clip': 0.2,
    'gamma': 0.99,
    'lamda': 0.97,
    'learning_rate': 0.01,
    'n_episode': 5000,
    'hidden_dim': 128,
    'memory_type': 'os',
    'rnn': False,
    'layer_norm': True,
    'minibatch_size': 128,
    'ppo_epoch': 10,
    'lossvalue_norm': True,
    'loss_coeff_value': 0.5,
    'loss_coeff_entropy': 0.01
}

DDPG = {
    'mamory_size': 100000,
    'alg': 'QLearning',
    'soft': False,
    'gamma': 0.99,
    'learning_rate': 0.001,
    'n_episode': 5000,
    'hidden_dim': 64,
    'batch_size': 128,
    'memory_type': 'os',
    'tau': 0.005,
    'epsilon_start': 0.5,
    'epsilon_end': 0.01,
    'time_length': 40000,
}

DDPG = {
    'mamory_size': 100000,
    'alg': 'QLearning',
    'soft': False,
    'gamma': 0.99,
    'learning_rate': 0.001,
    'n_episode': 5000,
    'hidden_dim': 64,
    'batch_size': 128,
    'memory_type': 'os',
    'tau': 0.005,
    'pi_target_update_interval': 2,
    'epsilon_start': 0.5,
    'epsilon_end': 0.01,
    'time_length': 40000,
}


PARAM['REINFORCE'] = REINFORCE
PARAM['QLearning'] = QLearning
PARAM['REINFORCE-B'] = REINFORCE_B
PARAM['QAC'] = QAC
PARAM['QAC-SN'] = QAC_SN
PARAM['OS-AC'] = OS_AC
PARAM['QAC-rnn'] = QAC_rnn
PARAM['PPO2'] = PPO2



