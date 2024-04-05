# An example ENN solution for the multi-armed bandit problem

import random

from typing import Dict, Mapping, List, Tuple
from entity_gym.runner import CliRunner
from entity_gym.env import *

# The `Environment` class defines the interface that all entity gym environments must implement.
class MultiArmedBandit(Environment):
    arms: List[Tuple[float]] = [(0.1,), (0.2,), (0.3,),]

    # The `obs_space` specifies the shape of observations returned by the environment.
    def obs_space(self) -> ObsSpace:
        return ObsSpace(
            entities={
                "arm": Entity(features=["reward"]),
                "player": Entity(features=[]),
            }
        )

    # The `action_space` specifies what actions that can be performed by the agent.
    def action_space(self) -> Dict[str, ActionSpace]:
        return {
            "pull": SelectEntityActionSpace(),
        }

    # `reset` should initialize the environment and return the initial observation.
    def reset(self) -> Observation:
        return self._observe()

    # `act` performs the chosen actions and returns the new observation.
    def act(self, actions: Mapping[ActionName, Action]) -> Observation:
        chosen_arm = actions["pull"].actees[0]
        reward = self.arms[chosen_arm][0]
        done = random.random() <= 0.2
        return self._observe(reward, done)

    def _observe(self, reward: float = 0.0, done: bool = False) -> Observation:
        return Observation(
            features={
                "arm": self.arms,
                "player": [[]],
            },
            ids={
                "player": [3],
                "arm": [0, 1, 2],
            },
            done=done,
            reward=reward,
            actions={
                "pull": SelectEntityActionMask(actor_types=["player"], actee_types=["arm"]),
            }
        )

from enn_trainer import TrainConfig, State, init_train_state, train
import hyperstate

@hyperstate.stateful_command(TrainConfig, State, init_train_state)

def main(state_manager: hyperstate.StateManager) -> None:
    train(state_manager=state_manager, env=MultiArmedBandit)

# if __name__ == "__main__":
#    main()

if __name__ == "__main__":
    env = MultiArmedBandit()
    # The `CliRunner` can run any environment with a command line interface.

    CliRunner(env).run()