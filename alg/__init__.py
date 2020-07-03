from .REINFORCE import REINFORCELearner
from .REINFORCE_baseline import REINFORCELearner as REINFORCE_baseline
from .AC import ACLearner

from .Qlearning import QLearner

Learner = {
    'REINFORCE': REINFORCELearner,
    'QLearning': QLearner,
    'REINFORCE-B': REINFORCE_baseline,
    'AC': ACLearner,
}