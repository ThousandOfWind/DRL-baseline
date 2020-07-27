from .REINFORCE import REINFORCELearner
from .REINFORCE_baseline import REINFORCELearner as REINFORCE_baseline
from .AC import ACLearner
from .QAC import ACLearner as QAC
from .QAC_sharednet import ACLearner as QAC_SN

from .oneStep_AC import ACLearner as OS_AC
from .PPO2 import PPOLearner


from .Qlearning import QLearner
from .SAC_discrete import SAC_Discrete

Learner = {
    'REINFORCE': REINFORCELearner,
    'QLearning': QLearner,
    'REINFORCE-B': REINFORCE_baseline,
    'AC': ACLearner,
    'QAC': QAC,
    'QAC-SN': QAC_SN,
    'OS-AC': OS_AC,
    'PPO2': PPOLearner,
    'SAC-discrete': SAC_Discrete
}