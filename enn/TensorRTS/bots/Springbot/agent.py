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
    def __init__(self, initial_observation: Observation, action_space: Dict[str, CategoricalActionSpace or SelectEntityActionSpace or GlobalCategoricalActionSpace], script_dir: str = ""):
        super().__init__(initial_observation, action_space, script_dir)
        self.tree = mcts(timeLimit=300)
        self.simgame = Interactive_TensorRTS(enable_printouts=False, trace_file="")
        # may need to change the MCTS implementation if progress capture doesnt work correctly...
    def on_game_start(self, is_player_one : bool, is_player_two : bool) -> None:
        return super().on_game_start(is_player_one, is_player_two)

    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    
    def take_turn(self, current_game_State : Observation) -> Mapping[ActionName, Action]:
        # tree = mcts(timeLimit=300)
        # now have once instance we update instead of recreating one each turn
        try:
            # pick an action by searching the tree from the current gamestate
            action_choice = tree.search(initialState=MCTS_State(current_game_State, list(enumerate(self.action_space["Move"].index_to_label)), self.simgame, self.is_player_two, list(map(lambda x: x[0], current_game_State.features["Tensor"])) ))
            # make the tree root for next time be the node corresponding to the action selected.
            tree.root = (node for action, node in self.root.children.items() if action is action_choice).__next__()
        except:
            # if an error occurs while selecting a move, pick a random one
            action_choice = random.choice(list(enumerate(self.action_space["Move"].index_to_label)))

        # print(action_choice)

        # format move correctly and return it
        return map_move(action_choice[0], action_choice[1])

# extension of the obervation class, adds additional functions so it can be used with MCTS
class MCTS_State(Observation):
    def __init__(self, state: Observation, action_space:  list, simgame: Interactive_TensorRTS, is_player_two: bool = False, initial_tensor_positions = [0,0]):
        self.state = state
        self.action_space = action_space
        self.is_player_two = is_player_two
        self.initial_tensor_positions = initial_tensor_positions
        self._game = simgame # make a game to simulate within.
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
        # inefficient, but has to work around the observation state not having a takeAction function
        # sets the simulation game state to the current gamestate, then acts within it - once as this player, once using a random opponent move.
        # then returns the state that results.
        self._game.set_state(self.state, self.initial_tensor_positions)
        self._game.act(map_move(action[0], action[1]), False, self.is_player_two, False) # take the action specified
        opp_rand_move = random.choice(self.action_space) # pick a random move for the opponent to do
        return MCTS_State(self._game.act(map_move(opp_rand_move[0], opp_rand_move[1]), False, not self.is_player_two, False), self.action_space, self._game, self.is_player_two, self.initial_tensor_positions) # perform the opponent's move and return the resulting state.
    def isTerminal(self):
        return self.state.done
    def getReward(self):
        return self.state.reward


def map_move(idx : int , label : str):
    action_map = {}
    action_map["Move"] = GlobalCategoricalAction(idx, label)
    return action_map

def flip_board(state : Observation) -> Observation:
    mapsize = 32
    ## credit to James Pennington for the reversal implementation
    ## https://github.com/jp013718/TensorRTS_Selfplay 
    opp_clusters = [[mapsize - i - 1, j] for i,j in state.features["Cluster"]]
    for cluster in opp_clusters:
        cluster[0] = mapsize - cluster[0] - 1
    opp_tensors = [[mapsize - i - 1, j, k, l] for i,j,k,l in state.features["Tensor"]]
    for tensor in opp_tensors:
        tensor[0] = mapsize - tensor[0] - 1
    return Observation(
        entities={
            "Cluster": (
                opp_clusters,
                [("Cluster", i) for i in range(len(opp_clusters))],
            ),
            "Tensor": (
                opp_tensors,
                [("Tensor", i) for i in range(len(opp_tensors))],
            ),
        },
        actions={
            "Move": GlobalCategoricalActionMask(),
        },
        done=state.done,
        reward=state.reward,
    )

def agent_hook(init_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir: str) -> Agent: 
    """Creates an agent of this type

    Returns:
        Agent: _description_
    """
    return New_Agent(init_observation, action_space, script_dir)

def display_name_hook() -> str: 
    """Provide the name of the student as a string

    Returns:
        str: Name of student
    """
    return 'Silas Springer'


if __name__ == "__main__":  # This is to run wth agents
    print(tensor_path)
    runner = GameRunner(enable_printouts=True, trace_file="")
    # init_observation = runner.set_new_game()
    init_observation = runner.game.observe()
    agent1 = New_Agent(init_observation, runner.game.action_space(), script_dir=".")
    agent2 = Random_Agent(init_observation, runner.game.action_space())

    runner.assign_players(agent1, agent2)
    runner.run(max_game_turns = 300)
