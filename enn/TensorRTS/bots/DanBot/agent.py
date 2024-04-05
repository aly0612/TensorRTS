import sys
import os
import re
tensor_path = os.path.abspath(os.path.join(os.path.basename(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(tensor_path)
sys.path.append("../..")  # Add the grandparent directory

import random 
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *
from TensorRTS import Agent
from enn_trainer import load_checkpoint, RogueNetAgent

class DanBot(Agent):

    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir : str) -> None: 
        super().__init__(init_observation, action_space, script_dir)
        bot_dir = ""
        for dir in os.listdir(f"{script_dir}/checkpoint"):
            if re.compile(r"latest-step[0-9]{12}").match(dir):
                bot_dir = dir
        checkpoint = load_checkpoint(f"{script_dir}/checkpoint/{bot_dir}")
        self.current = RogueNetAgent(checkpoint.state.agent)

    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]:
        action, _ = self.current.act(current_game_state)
        return action
    
    def on_game_start(self, is_player_one : bool, is_player_two : bool) -> None:
        return super().on_game_start(is_player_one, is_player_two)
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    
def agent_hook(init_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir : str) -> Agent: 
    return DanBot(init_observation, action_space, script_dir)

def display_name_hook() -> str: 
    return 'Danny Nagura'

if __name__ == "__main__":  # This is to run wth a random agent
    from TensorRTS import GameRunner

    print("### Playing a game session with a random agent ... ")
    runner = GameRunner(enable_printouts=True, trace_file="GameTraceDanBot.txt")
    init_observation = runner.get_game_observation(is_player_two= False)
    agent = DanBot(init_observation, runner.game.action_space(), os.getcwd())

    runner.assign_players(agent)
    runner.run(max_game_turns = 300)
