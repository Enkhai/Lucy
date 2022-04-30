import pathlib

from stable_baselines3 import PPO

from rlgym_tools.extra_action_parsers.kbm_act import KBMAction


class Agent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        # Resolve model loading issues - dummy values,
        custom_objects = {
            # pickle in Python 3.7 cannot parse stored schedule functions, we specify custom objects
            'lr_schedule': 1e-4,
            'clip_range': .2,
            'n_envs': 1,  # need to specify one environment
        }

        self.actor = PPO.load(str(_path) + '/model_599040000_steps.zip', device='cpu', custom_objects=custom_objects)
        self.actor.policy.set_training_mode(False)
        self.parser = KBMAction()

    def act(self, state):
        action = self.actor.predict(state, deterministic=True)
        x = self.parser.parse_actions(action[0], state)

        return x[0]


if __name__ == "__main__":
    print("You're doing it wrong.")
