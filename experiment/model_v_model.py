import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
from lucy_utils.multi_instance_utils import get_match
from lucy_utils.obs import NectoObs
from lucy_utils.obs.old import OldGraphAttentionObs
from rlgym.utils.gamestates import GameState
from rlgym.utils.gamestates import PlayerData
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions import common_conditions
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3 import PPO

from lucy_match_params import LucyAction

# import deprecated `utils` package for trained Necto to work
utils_path = str(Path.home()) + "\\rocket_league_utils\\old_deprecated_utils"
sys.path.insert(0, utils_path)


class MultiModelObs(ObsBuilder):

    def __init__(self, obss: List[ObsBuilder], num_obs_players: List[int]):
        super(MultiModelObs, self).__init__()
        assert len(obss) == len(num_obs_players), "`obss` and `num_obs_players` lengths must match"
        self.obss = obss
        self.num_obs_players = np.cumsum(num_obs_players)
        self.p_idx = 0
        self.curr_state = None
        self.autodetect = True

    def reset(self, initial_state: GameState):
        [o.reset(initial_state) for o in self.obss]

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if self.autodetect:
            self.autodetect = False
            return np.zeros(0)

        if self.curr_state != state:
            self.p_idx = 0
            self.curr_state = state

        obs_idx = (self.p_idx >= self.num_obs_players).sum()
        self.p_idx += 1

        return self.obss[obs_idx].build_obs(player, state, previous_action)


if __name__ == '__main__':
    team_size = 2
    tick_skip = 8
    fps = 120 // tick_skip
    terminal_conditions = [common_conditions.TimeoutCondition(fps * 300),
                           common_conditions.NoTouchTimeoutCondition(fps * 45),
                           common_conditions.GoalScoredCondition()]
    # obs_builder = MultiModelObs([OldGraphAttentionObs(stack_size=5), NectoObs()], [2, 2])
    obs_builder = MultiModelObs([OldGraphAttentionObs(stack_size=5)], [4])

    match = get_match(reward=DefaultReward(),
                      terminal_conditions=terminal_conditions,
                      obs_builder=obs_builder,
                      action_parser=LucyAction(),
                      state_setter=DefaultState(),
                      team_size=team_size,
                      )
    env = SB3MultipleInstanceEnv([match])

    custom_objects = {
        # arbitrary
        'lr_schedule': 1e-4,
        'clip_range': .2,
        # 2v2
        'n_envs': 2,
    }

    blue_model = PPO.load("../models_folder/Perceiver_LucyReward_v4/model_2000000000_steps.zip",
                          device="cpu", custom_objects=custom_objects)
    orange_model = PPO.load("../models_folder/Perceiver_LucyReward_v3/model_2000000000_steps.zip",
                            device="cpu", custom_objects=custom_objects)

    max_score_count = 600

    match_name = "v4 vs v3, " + str(max_score_count) + " goals, 2 billion"

    blue_score_sum = 0
    orange_score_sum = 0
    blue_scores = []
    orange_scores = []
    while True:
        obs = env.reset()
        done = [False]

        while not done[0]:
            action = np.concatenate((blue_model.predict(np.stack(obs[:2]))[0],
                                     orange_model.predict(np.stack(obs[2:]))[0]))[None]
            obs, reward, done, gameinfo = env.step(action)

        final_state: GameState = gameinfo[0]['state']

        blue_score_dif = final_state.blue_score - blue_score_sum
        orange_score_dif = final_state.orange_score - orange_score_sum

        blue_score_sum += blue_score_dif
        orange_score_sum += orange_score_dif

        blue_scores.append(blue_score_sum)
        orange_scores.append(orange_score_sum)

        if blue_score_sum >= max_score_count or orange_score_sum >= max_score_count:
            break

    df = pd.DataFrame([blue_scores, orange_scores]).T
    df.columns = ["Blue score", "Orange score"]
    df.to_csv("evaluation_results/" + match_name + ".csv")

    print("\n\n")
    print("====================")
    print("RESULT")
    print("====================")
    print("Blue:", blue_score_sum)
    print("Orange:", orange_score_sum)

    env.close()
