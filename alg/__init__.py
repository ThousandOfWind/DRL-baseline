from .REINFORCE import REINFORCELearner
from .REINFORCE_baseline import REINFORCELearner as REINFORCE_baseline
from .AC import ACLearner
from .QAC import ACLearner as QAC
from .QAC_sharednet import ACLearner as QAC_SN


from .Qlearning import QLearner

Learner = {
    'REINFORCE': REINFORCELearner,
    'QLearning': QLearner,
    'REINFORCE-B': REINFORCE_baseline,
    'AC': ACLearner,
    'QAC': QAC,
    'QAC-SN': QAC_SN
}