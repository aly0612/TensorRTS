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

    def observe(self) -> Observation:
        return Observation(
            global_features=[self.x_pos, self.y_pos], done=False, reward=0
        )

    def act(self, actions: Mapping[ActionName, Action]) -> Observation:
        return self.observe()

    def action_space(self) -> Dict[str, ActionSpace]:
        return {}

if __name__ == "__main__":
    env = TreasureHunt()
    # The `CliRunner` can run any environment with a command line interface.
    CliRunner(env).run()