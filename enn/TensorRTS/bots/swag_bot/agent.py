import os
import sys
sys.path.append("../..")  # Add the grandparent directory

import random 
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *
from TensorRTS import Agent
from enn_trainer import load_checkpoint, RogueNetAgent

class SwagBot(Agent):
    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir : str) -> None: 
        super().__init__(init_observation, action_space, script_dir)
        checkpoint = load_checkpoint(f"{script_dir}/checkpoint")
        self.current = RogueNetAgent(checkpoint.state.agent)

    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]:
        return self.current.act(current_game_state)[0]
    
    def on_game_start(self, is_player_one : bool, is_player_two : bool) -> None:
        return super().on_game_start(is_player_one, is_player_two)
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    
def agent_hook(init_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir) -> Agent: 
    return SwagBot(init_observation, action_space, script_dir)

def display_name_hook() -> str: 
    return 'Jansen'

if __name__ == "__main__":
  print("\n---Testing Swag Bot---\n")
  from TensorRTS import GameRunner

  cwd = os.getcwd()
  runner = GameRunner(enable_printouts=True, trace_file="SwagTrace.txt")
  init_observation = runner.get_game_observation(is_player_two=False)
  swag_bot = SwagBot(init_observation, runner.game.action_space(),cwd)

  runner.assign_players(swag_bot)
  runner.run(max_game_turns=300)
