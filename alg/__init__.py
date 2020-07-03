from .REINFORCE import REINFORCELearner
from .Qlearning import QLearner

Learner = {
    'REINFORCE': REINFORCELearner,
    'QLearning': QLearner,
}