import sys
import os

tensor_path = os.path.abspath(os.path.join(os.path.basename(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(tensor_path)

import random 
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import ActionName, Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *
from TensorRTS import Agent, Interactive_TensorRTS, GameRunner, TensorRTS, Random_Agent

    
from mcts import mcts

class New_Agent(Agent):
    def __init__(self, initial_observation: Observation, action_space: Dict[str, CategoricalActionSpace or SelectEntityActionSpace or GlobalCategoricalActionSpace], script_dir: str):
        super().__init__(initial_observation, action_space, script_dir)
    
    def on_game_start(self, is_player_one : bool, is_player_two : bool) -> None:
        return super().on_game_start(is_player_one, is_player_two)
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    
    def take_turn(self, current_game_State : Observation) -> Mapping[ActionName, Action]:
        tree = mcts(timeLimit=1000)
        try:
            action_choice = tree.search(initialState=MCTS_State(current_game_State, list(enumerate(self.action_space["Move"].index_to_label)) ))
        except:
            action_choice = random.choice(list(enumerate(self.action_space["Move"].index_to_label)))
        return map_move(action_choice[0], action_choice[1])

class MCTS_State(Observation):
    def __init__(self, state: Observation, action_space:  list):
        self.state = state
        self.action_space = action_space
    def __copy__(self):
        return MCTS_State(self.state.copy(), self.action_space)
    def __deepcopy__(self, memo):
        return MCTS_State(self.state.deepcopy(), self.action_space)
    def getPossibleActions(self):
        actionmask = self.state.actions["Move"].mask
        if actionmask is None:
            return self.action_space
        return list(filter(lambda x, i: actionmask[i] == True, enumerate(self.action_space)))
    def takeAction(self, action):
        _game = Interactive_TensorRTS(enable_printouts=False)
        _game.set_state(self.state)
        return MCTS_State(_game.act(map_move(action[0], action[1])), self.action_space)
    def isTerminal(self):
        return self.state.done
    def getReward(self):
        done = self.tensors[0][0] >= self.tensors[1][0]
        if done:
            reward = 25 if self.tensor_power(0) > self.tensor_power(1) else 0 if self.tensor_power(0) == self.tensor_power(1) else -25
        else:
            reward = 0
            # Shaped reward for early training
            for cluster in self.clusters:
                if (self.tensors[0][0] - cluster[0] < 3 and self.tensors[0][0] - cluster[0] > -3):
                    reward += 0.01 * cluster[1]
                elif (self.tensors[0][0] - cluster[0] < 2 and self.tensors[0][0] - cluster[0] > -2):
                    reward += 0.02 * cluster[1]
            reward += min(10,((0.05 * self.tensors[0][2]) + (0.15 * self.tensors[0][3])))
            # Factor in whether the tensor is in danger (opponent is close and has more power) or has advantage (opponent is close and has less power)
            if self.tensors[1][0] - self.tensors[0][0] < 2 and self.tensors[1][0] - self.tensors[0][0] > -2:
                if self.tensor_power(1) > self.tensor_power(0):
                    reward -= 5
                elif self.tensor_power(1) < self.tensor_power(0):
                    reward += 5
            elif self.tensors[1][0] - self.tensors[0][0] < 3 and self.tensors[1][0] - self.tensors[0][0] > -3:
                if self.tensor_power(1) > self.tensor_power(0):
                    reward -= 2.5
                elif self.tensor_power(1) < self.tensor_power(0):
                    reward += 2.5
            elif self.tensors[1][0] - self.tensors[0][0] < 4 and self.tensors[1][0] - self.tensors[0][0] > -4:
                if self.tensor_power(1) > self.tensor_power(0):
                    reward -= 1
                elif self.tensor_power(1) < self.tensor_power(0):
                    reward += 1
        return reward

def map_move(idx : int , label : str):
    action_map = {}
    action_map["Move"] = GlobalCategoricalAction(idx, label)
    return action_map

def agent_hook(init_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir: str) -> Agent: 
    """Creates an agent of this type

    Returns:
        Agent: _description_
    """
    # return Random_Agent(init_observation, action_space)
    return New_Agent(init_observation, action_space, script_dir)

def display_name_hook() -> str: 
    """Provide the name of the student as a string

    Returns:
        str: Name of student
    """
    return 'Kaleb Demaline'

if __name__ == "__main__":  # This is to run wth agents
    print(tensor_path)
    runner = GameRunner(enable_printouts=True, trace_file="")
    init_observation = runner.game.observe()
    agent1 = New_Agent(init_observation, runner.game.action_space())
    agent2 = Random_Agent(init_observation, runner.game.action_space())

    runner.assign_players(agent1, agent2)
    runner.run(max_game_turns = 300)
