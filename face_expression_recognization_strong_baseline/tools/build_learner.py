from tools.simple_learner import SimpleLearner
from tools.scn_learner import ScnLearner
# from tools.scn_learner_for_infer import ScnLearner
from tools.ran_learner import RanLearner


def build_learner(cfg, **kwargs):
    learner_name = cfg.TRAIN.learner
    if learner_name == 'simple':
        return SimpleLearner(**kwargs)

    elif learner_name == 'scn':
        return ScnLearner(**kwargs)

    elif learner_name == 'ran':
        return RanLearner(**kwargs)