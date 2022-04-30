from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

import numpy as np
from agent import Agent
from obs.graph_attention_obs import GraphAttentionObs
from rlgym_compat import GameState


class RLGymExampleBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        self.obs_builder = GraphAttentionObs(stack_size=5)  # adjustable
        self.agent = Agent()  # neural network logic
        self.tick_skip = 8  # depends on the tick skip bot was trained on

        self.game_state = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        print(f'{self.name} Ready - Index:', index)

    def initialize_agent(self):
        # Initialize the rlgym GameState object now that the game is active and the info is available
        self.game_state = GameState(self.get_field_info())
        self.ticks = self.tick_skip  # So we take an action for the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.update_action = True

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # Update time
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        # Decode game state
        ticks_elapsed = round(delta * 120)  # multiply by physics engine fps and round
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        # If enough ticks have elapsed and the agent is eligible for action
        if self.update_action and len(self.game_state.players) > self.index:
            if packet.game_info.is_kickoff_pause and not packet.game_info.is_round_active:
                # Reset if kickoff because we use a stacking obs
                self.obs_builder.reset(self.game_state)
                self.action = np.zeros(8)
                self.update_controls(self.action)

            # Build the gamestate players
            player = self.game_state.players[self.index]
            teammates = [p for p in self.game_state.players if p.team_num == self.team and p != player]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]

            self.game_state.players = [player] + teammates + opponents

            obs = self.obs_builder.build_obs(player, self.game_state, self.action)  # Build the observation
            self.action = self.agent.act(obs)  # and get the action

            self.update_action = False  # No more need to update action

        # If it's time to act (determined by the tick skip), update agent controls
        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_controls(self.action)
            self.update_action = True

        return self.controls

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = 0 if action[5] > 0 else action[3]  # *dodge fix
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0


if __name__ == "__main__":
    print("You're doing it wrong.")
