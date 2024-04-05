import sys
import os
import random 
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *
from TensorRTS import Agent, GameRunner, Random_Agent
from enn_trainer import load_checkpoint, RogueNetAgent


tensor_path = os.path.abspath(os.path.join(os.path.basename(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(tensor_path)

class skywill_bot(Agent):
    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir) -> None: 
        super().__init__(init_observation, action_space, script_dir)
        self.move_counter = 0


    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]:
        mapping = {}

        moves_sequence = ["advance", "retreat", "rush", "boom"]
        move_label = moves_sequence[self.move_counter]
        move_index = self.move_counter
        self.move_counter = (self.move_counter + 1) % len(moves_sequence)
        mapping['Move'] = GlobalCategoricalAction(move_index, move_label)

        return mapping
    
    def on_game_start(self, is_player_one : bool, is_player_two : bool) -> None:
        return super().on_game_start(is_player_one, is_player_two)
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    

def agent_hook(init_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir) -> Agent: 
    return skywill_bot(init_observation, action_space, script_dir)

def display_name_hook() -> str: 
    return 'Osborn Koranteng'


if __name__ == "__main__":
    runner = GameRunner(enable_printouts = True)
    init_observation = runner.get_game_observation(is_player_two=False)
    skywill = skywill_bot(init_observation, runner.game.action_space())
    random_agent = Random_Agent(init_observation, runner.game.action_space())
    runner.assign_players(skywill, random_agent)
    runner.run(max_game_turns=300)


