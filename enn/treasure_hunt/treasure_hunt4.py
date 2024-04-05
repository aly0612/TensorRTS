# https://entity-gym.readthedocs.io/en/latest/quick-start-guide.html

import random
from typing import Mapping, Tuple, Dict
from entity_gym.runner import CliRunner
from entity_gym.env import *

# The `Environment` class defines the interface that all entity gym environments must implement.
class TreasureHunt(Environment):
    def reset(self) -> Observation:
        self.x_pos = 0
        self.y_pos = 0
        self.game_over = False
        self.traps = []
        self.treasure = []
        for _ in range(5):
            self.traps.append(self._random_empty_pos())
        for _ in range(5):
            self.treasure.append(self._random_empty_pos())
        return self.observe()

    def obs_space(self) -> ObsSpace:
        return ObsSpace(
            global_features=["x_pos", "y_pos"],
            # An observation space can have several entities with different features.
            # On any given step, an observation may include any number of the defined entities.
            entities={
                "Trap": Entity(features=["x_pos", "y_pos"]),
                "Treasure": Entity(features=["x_pos", "y_pos"]),
            }
        )

    def action_space(self) -> Dict[str, ActionSpace]:
        # The `GlobalCategoricalActionSpace` allows the agent to choose from set of discrete actions.
        return {
            "move": GlobalCategoricalActionSpace(["up", "down", "left", "right"])
        }
    
    def act(self, actions: Mapping[ActionName, Action]) -> Observation:
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

        reward = 0.0
        if (self.x_pos, self.y_pos) in self.treasure:
            reward = 1.0
            self.treasure.remove((self.x_pos, self.y_pos))
        if (self.x_pos, self.y_pos) in self.traps or len(self.treasure) == 0:
            self.game_over = True

        return self.observe(reward)

    def observe(self, reward: float = 0.0) -> Observation:
        return Observation(
            global_features=[self.x_pos, self.y_pos],
            features={
                "Trap": self.traps,
                "Treasure": self.treasure,
            },
            done=self.game_over,
            reward=reward,
            actions={"move": GlobalCategoricalActionMask()},
        )

    def _random_empty_pos(self) -> Tuple[int, int]:
        # Generate a random position on the grid that is not occupied by a trap, treasure, or player.
        while True:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            if (x, y) not in (self.traps + self.treasure + [(self.x_pos, self.y_pos)]):
                return x, y

if __name__ == "__main__":
    env = TreasureHunt()
    # The `CliRunner` can run any environment with a command line interface.
    CliRunner(env).run()