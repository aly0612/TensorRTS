import random 
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *
from enn_trainer import load_checkpoint, RogueNetAgent
from TensorRTS import Agent

class S2OPBot(Agent):
    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir : str) -> None: 
        super().__init__(init_observation, action_space, script_dir)
        last = load_checkpoint(f'{script_dir}/checkpoints')
        self.agent = RogueNetAgent(last.state.agent)
        
    def take_turn(self, current_game_state : Observation) -> Action:
        if self.is_player_one:
            action, predicted_return = self.agent.act(current_game_state)
            return action
        elif self.is_player_two:
            entities = current_game_state.features
            actions = current_game_state.actions
            done = current_game_state.done
            reward = current_game_state.reward

            clusters = entities["Cluster"]
            tensors = entities["Tensor"]

            opp_clusters = [[32-i-1, j] for i, j in clusters]
            opp_tensors = [[32-i-1, j, k, l] for i, j, k, l in tensors]
            opp_obs = Observation(
            entities={
            "Cluster": (
                opp_clusters,
                [("Cluster", i) for i in range(len(opp_clusters))]
            ),
            "Tensor": (
                opp_tensors,
                [("Tensor", i) for i in range(len(opp_tensors))]
            )
            },
            actions=actions,
            done=done,
            reward=reward
            )

            action, predicted_return = self.agent.act(opp_obs)
            return action
    
    def on_game_start(self, is_player_one : bool, is_player_two : bool) -> None:
        return super().on_game_start(is_player_one, is_player_two)
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    
def agent_hook(init_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir) -> Agent: 
    return S2OPBot(init_observation, action_space, script_dir)
    
def display_name_hook() -> str: 
    """Provide the name of the student as a string

    Returns:
        str: Name of student
    """
    return 'Owen Salyer'
