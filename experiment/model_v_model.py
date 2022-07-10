import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import rlgym
from lucy_utils.obs import NectoObs
from rlgym.utils.gamestates import GameState
from rlgym.utils.gamestates import PlayerData
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.terminal_conditions import common_conditions
from stable_baselines3 import PPO

from lucy_match_params import LucyObs, LucyAction

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
    # obs_builder = MultiModelObs([LucyObs(stack_size=5), NectoObs()], [2, 2])
    obs_builder = MultiModelObs([LucyObs(stack_size=5)], [4])

    env = rlgym.make(game_speed=100,
                     tick_skip=tick_skip,
                     self_play=True,
                     team_size=team_size,
                     terminal_conditions=terminal_conditions,
                     obs_builder=obs_builder,
                     action_parser=LucyAction(),
                     )

    custom_objects = {
        # arbitrary
        'lr_schedule': 1e-4,
        'clip_range': .2,
        # 2v2
        'n_envs': 2,
    }

    blue_model = PPO.load("../models_folder/Perceiver_LucyReward_v3/model_1500800000_steps.zip",
                          device="cpu", custom_objects=custom_objects)
    orange_model = PPO.load("../models_folder/Perceiver_LucyReward_v3_batch_12800/model_2000000000_steps.zip",
                            device="cpu", custom_objects=custom_objects)

    max_score_count = 500

    blue_score = 0
    orange_score = 0

    score_count = 0
    while True:
        obs = env.reset()
        done = False

        while not done:
            action = np.concatenate((blue_model.predict(obs[:2])[0], orange_model.predict(obs[2:])[0]))
            obs, reward, done, gameinfo = env.step(action)

        final_state: GameState = gameinfo['state']

        blue_score_dif = final_state.blue_score - blue_score
        orange_score_dif = final_state.orange_score - orange_score

        score_count += blue_score_dif + orange_score_dif
        blue_score += blue_score_dif
        orange_score += orange_score_dif

        if score_count >= max_score_count:
            break

    print("\n\n")
    print("====================")
    print("RESULT")
    print("====================")
    print("Blue:", blue_score)
    print("Orange:", orange_score)

    env.close()
