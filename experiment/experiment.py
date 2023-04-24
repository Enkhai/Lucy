from lucy_utils.algorithms import DeviceAlternatingPPO
from lucy_utils.models import PerceiverNet
from lucy_utils.multi_instance_utils import config, get_matches
from lucy_utils.obs import NextoObsBuilder
from lucy_utils.parsers import NextoAction
from lucy_utils.policies import ActorCriticAttnPolicy
from lucy_utils.rewards.nexto_reward import NextoRewardFunction
from lucy_utils.rewards.sb3_log_reward import SB3NamedLogRewardCallback
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.sb3_utils.sb3_instantaneous_fps_callback import SB3InstantaneousFPSCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from lucy_match_params import LucyTerminalConditions, LucyState

models_folder = "models_folder/"
tensorboard_log_dir = "bin"
model_name = "Nexto_Perceiver"

if __name__ == '__main__':
    # ----- ENV CONFIG -----

    num_instances = 1
    agents_per_match = 2 * 2  # self-play
    n_steps, batch_size, gamma, fps, save_freq = config(num_instances=num_instances,
                                                        avg_agents_per_match=agents_per_match,
                                                        target_steps=320_000,
                                                        target_batch_size=4_000,
                                                        callback_save_freq=10)

    action_stacking = 1

    matches = get_matches(reward_cls=NextoRewardFunction,
                          terminal_conditions=lambda: LucyTerminalConditions(fps),
                          obs_builder_cls=NextoObsBuilder,
                          action_parser_cls=NextoAction,
                          state_setter_cls=LucyState,
                          sizes=[agents_per_match // 2] * num_instances  # self-play, hence // 2
                          )

    # ----- ENV SETUP -----

    env = SB3MultipleInstanceEnv(match_func_or_matches=matches)
    env = VecMonitor(env)

    # ----- MODEL SETUP -----

    critic_net_arch = dict(
        # minus one for the key padding mask
        query_dims=env.observation_space.shape[-1] - 1,
        # minus the stack for the previous actions
        kv_dims=env.observation_space.shape[-1] - 1 - (action_stacking * 8),
        # the rest is default arguments
    )
    actor_net_arch = dict(critic_net_arch)
    actor_net_arch['player_emb_net_shape'] = [32]
    actor_net_arch['action_emb_net_shape'] = [32] * 3

    policy_kwargs = dict(network_classes=PerceiverNet,
                         net_arch=[actor_net_arch, critic_net_arch],
                         action_stack_size=action_stacking,
                         is_nexto=True,
                         # use_rp=True,
                         # use_sr=True,
                         # rp_seq_len=20,
                         # zero_rew_threshold=0.009
                         )

    # model = DeviceAlternatingPPO.load("./models_folder/Perceiver/model_743680000_steps.zip", env)
    model = DeviceAlternatingPPO(policy=ActorCriticAttnPolicy,
                                 env=env,
                                 learning_rate=1e-4,
                                 n_steps=n_steps,
                                 gamma=gamma,
                                 batch_size=batch_size,
                                 tensorboard_log=tensorboard_log_dir,
                                 policy_kwargs=policy_kwargs,
                                 verbose=1,
                                 )

    # ----- TRAINING -----

    callbacks = [SB3InstantaneousFPSCallback(),
                 SB3NamedLogRewardCallback(),
                 CheckpointCallback(save_freq,
                                    save_path=models_folder + model_name,
                                    name_prefix="model"),
                 ]

    model.learn(total_timesteps=3_500_000_000,
                callback=callbacks,
                tb_log_name=model_name,
                # reset_num_timesteps=False
                )

    # ----- CLOSE -----

    model.save(models_folder + model_name + "_final")
    env.close()
