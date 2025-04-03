from enum import Enum

class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    GRPO_MEAN_SUBTRACTION = 'grpo_mean_subtraction'
    GRPO_NO_NORMALIZATION = 'grpo_no_normalization'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'
