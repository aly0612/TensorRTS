import os

import sys
sys.path.append("../..")

from TensorRTS import Agent
from typing import Dict, Mapping
from entity_gym.env import Observation
from entity_gym.env import *
from enn_trainer import load_checkpoint, RogueNetAgent

class NeutronBot(Agent):
  def __init__(self, initial_observation: Observation, action_space: Dict[ActionName, ActionSpace], script_dir : str, bot_dir: str="neutron_checkpoint"):
    super().__init__(initial_observation, action_space, script_dir)
    self.script_dir = script_dir
    self.bot_dir = bot_dir
    self.kd_ratio = 0
    self.games = 0

  def take_turn(self, current_game_state: Observation) -> Mapping[ActionName, Action]:
    action, predicted_return = self.agent.act(current_game_state)
    return action
      
  def on_game_start(self, is_player_one:bool, is_player_two:bool) -> None:
    super().on_game_start(is_player_one, is_player_two)
    if is_player_one:
      checkpoint = load_checkpoint(os.path.join(self.script_dir, self.bot_dir, "player_1"))
      self.agent = RogueNetAgent(checkpoint.state.agent)
    elif is_player_two:
      checkpoint = load_checkpoint(os.path.join(self.script_dir, self.bot_dir, "player_2"))
      self.agent = RogueNetAgent(checkpoint.state.agent)
    return
  
  def on_game_over(self, did_i_win: bool, did_i_tie: bool) -> None:
    self.games += 1
    if did_i_win:
      self.kd_ratio = (self.kd_ratio*(self.games-1)+1)/self.games
    return super().on_game_over(did_i_win, did_i_tie)
  
def agent_hook(init_observation: Observation, action_space: Dict[ActionName, ActionSpace], script_dir : str) -> Agent:
  return NeutronBot(init_observation, action_space, script_dir)
  
def display_name_hook() -> str:
  return "James Pennington"

if __name__ == "__main__":
  print("\n---Testing Neutron Bot---\n")
  from TensorRTS import GameRunner

  cwd = os.getcwd()

  runner = GameRunner(enable_printouts=True, trace_file="NeutronTest.txt")
  init_observation = runner.get_game_observation(is_player_two=False)
  neutron_bot = NeutronBot(init_observation, runner.game.action_space(), script_dir=cwd)

  runner.assign_players(neutron_bot)
  runner.run(max_game_turns=300)
