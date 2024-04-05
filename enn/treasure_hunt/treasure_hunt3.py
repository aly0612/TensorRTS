from typing import Dict, Mapping
from entity_gym.runner import CliRunner
from entity_gym.env import *

# The `Environment` class defines the interface that all entity gym environments must implement.
class TreasureHunt(Environment):
    def obs_space(self) -> ObsSpace:
        # `global_features` adds a fixed-length vector of features to the observation.
        return ObsSpace(global_features=["x_pos", "y_pos"])

    def reset(self) -> Observation:
        self.x_pos = 0
        self.y_pos = 0
        return self.observe()

    def action_space(self) -> Dict[str, ActionSpace]:
        # The `GlobalCategoricalActionSpace` allows the agent to choose from set of discrete actions.
        return {
            "move": GlobalCategoricalActionSpace(["up", "down", "left", "right"])
        }

    def act(self, actions: Mapping[ActionName, Action]) -> Observation:
        # Adjust the player's position according to the chosen action.
        action = actions["move"]
        assert isinstance(action, GlobalCategoricalAction)
        if action.label == "up" and self.y_pos < 10:
            self.y_pos += 1
        elif action.label == "down" and self.y_pos > -10:
            self.y_pos -= 1
        elif action.label == "left" and self.x_pos > -10:
            self.x_pos -= 1
        elif action.label == "right" and self.x_pos < 10:
            self.x_pos += 1
        return self.observe()

    def observe(self) -> Observation:
        return Observation(
            global_features=[self.x_pos, self.y_pos],
            done=False,
            reward=0,
            # Each `Observation` must specify which actions are available on the current step.
            actions={"move": GlobalCategoricalActionMask()},
        )

if __name__ == "__main__":
    env = TreasureHunt()
    # The `CliRunner` can run any environment with a command line interface.
    CliRunner(env).run()