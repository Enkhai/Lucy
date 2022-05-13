from rlgym.utils.state_setters import DefaultState

from lucy_match_params import LucyReward, LucyTerminalConditions, LucyObs, LucyAction
from lucy_utils.load_evaluate import load_and_evaluate

if __name__ == '__main__':
    custom_objects = {
        # pickle in Python 3.7 cannot parse stored schedule functions, we specify custom objects
        'lr_schedule': 1e-4,
        'clip_range': .2,
        'n_envs': 1,  # need to specify one environment
    }

    kwargs = {'custom_objects': custom_objects}

    load_and_evaluate("../models_folder/NectoReward_ownPerceiver_2x128/NectoReward_ownPerceiver_2x128_final.zip",
                      2,
                      LucyTerminalConditions(15),
                      LucyObs(stack_size=5),
                      DefaultState(),  # we use the default state for evaluation
                      LucyAction(),
                      LucyReward(0.995),
                      kwargs=kwargs
                      )
