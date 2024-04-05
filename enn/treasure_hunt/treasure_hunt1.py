# https://entity-gym.readthedocs.io/en/latest/quick-start-guide.html
# See also: https://github.com/entity-neural-network/enn-template/blob/main/enn_template/env.py

from typing import Dict, Mapping
from entity_gym.runner import CliRunner
from entity_gym.env import *

# The `Environment` class defines the interface that all entity gym environments must implement.
class TreasureHunt(Environment):
    # The `obs_space` specifies the shape of observations returned by the environment.
    def obs_space(self) -> ObsSpace:
        return ObsSpace()

    # The `action_space` specifies what actions that can be performed by the agent.
    def action_space(self) -> Dict[str, ActionSpace]:
        return {}

    # `reset` should initialize the environment and return the initial observation.
    def reset(self) -> Observation:
        return Observation.empty()

    # `act` performs the chosen actions and returns the new observation.
    def act(self, actions: Mapping[ActionName, Action]) -> Observation:
        return Observation.empty()


if __name__ == "__main__":
    env = TreasureHunt()
    # The `CliRunner` can run any environment with a command line interface.
    CliRunner(env).run()
