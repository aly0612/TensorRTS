import sys
sys.path.append("../..")  # Add the grandparent directory



def shaped_rewards(tensors,tensor_power,clusters):
    # return 20
    done = tensors[0][0] >= tensors[1][0]
    reward = 0
    if done:
        reward = 25 if tensor_power(0) > tensor_power(1) else 0 if tensor_power(0) == tensor_power(1) else -25
    else:
        
        #Shaped reward for early training
        for cluster in clusters:
            if (tensors[0][0] - cluster[0] < 3 and tensors[0][0] - cluster[0] > -3):
                reward += 0.01 * cluster[1]
            elif (tensors[0][0] - cluster[0] < 2 and tensors[0][0] - cluster[0] > -2):
                reward += 0.02 * cluster[1]
        reward += min(20,((0.05 * tensors[0][2]) + (0.15 * tensors[0][3])))
        # Factor in whether the tensor is in danger (opponent is close and has more power) or has advantage (opponent is close and has less power)
        if tensors[1][0] - tensors[0][0] < 2 and tensors[1][0] - tensors[0][0] > -2:
            if tensor_power(1) > tensor_power(0):
                reward -= 5
            elif tensor_power(1) < tensor_power(0):
                reward += 5
        elif tensors[1][0] - tensors[0][0] < 3 and tensors[1][0] - tensors[0][0] > -3:
            if tensor_power(1) > tensor_power(0):
                reward -= 2.5
            elif tensor_power(1) < tensor_power(0):
                reward += 2.5
        elif tensors[1][0] - tensors[0][0] < 4 and tensors[1][0] - tensors[0][0] > -4:
            if tensor_power(1) > tensor_power(0):
                reward -= 1
            elif tensor_power(1) < tensor_power(0):
                reward += 1
    return reward


#some useful functions (these already exist withing numpy/torch, but i was too lazy to find them)
import functools
def flatten_dict_of_arrays(data, prefix=""):
  flattened_list = []
  def flatten_inner(value, key):
    if isinstance(value, list):
      for i, v in enumerate(value):
        flatten_inner(v, f"{prefix}{key}[{i}]")
    else:
      flattened_list.append((f"{prefix}{key}", value))

  for key, value in data.items():
    flatten_inner(value, key)

  return [value for _, value in flattened_list]


def value_to_discrete_4(input):
    output = [0,0,0,0]
    output[input] = 1
    return output

index_to_moves = {
    0 : "advance",
    1 : "retreat",
    2 : "rush",
    3 : "boom"
}
moves_to_index = {
    "advance": 0,
    "retreat": 1,
    "rush": 2,
    "boom": 3,
}

from TensorRTS import Agent, GameResult, Interactive_TensorRTS,GameRunner,Runtime_Failure_Exception
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import Observation

### custom game runner that saves states

class GameRunnerSaveStates(): 
    def __init__(self, environment = None, enable_printouts : bool = False, trace_file : str = ""):
        self.enable_printouts = enable_printouts
        self.game = Interactive_TensorRTS(enable_printouts=enable_printouts, trace_file=trace_file)
        self.game.reset()

        self.player_one = None
        self.player_two = None
        self.trace_file = trace_file
        self.results : GameResult = None

        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
    
    def assign_players(self, first_agent : Agent, second_agent : Agent = None, first_agent_student_name : str = None, second_agent_student_name : str = None):
        self.player_one = first_agent

        if second_agent is not None:
            self.player_two = second_agent
        # if self.trace_file:
        #     with open(self.trace_file, 'a') as file: 
        #         file.write(f'Player one: {first_agent_student_name}\n')
        #         file.write(f'Player two: {second_agent_student_name}\n')

    def flip_board(state : Observation) -> Observation: 
        mapsize = 32
        ## credit to Silas Springer and James Pennington for the reversal implementation
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
    
    def get_game_observation(self, is_player_two : bool) -> Observation: 
        if is_player_two:
            return GameRunner.flip_board(self.game.observe())
        return self.game.observe()
    
    def run(self, max_game_turns : int): 
        assert(self.player_one is not None)
        num_game_turns = 0
        
        game_state = self.game.observe()
        self.player_one.on_game_start(is_player_one=True, is_player_two=False)
        if self.player_two is not None: 
            self.player_two.on_game_start(is_player_one=True, is_player_two=False)

        while(self.game.is_game_over is False):
            if num_game_turns == max_game_turns: 
                break
            
            try:
                #take moves and pass updated environments to agents
                #MODIFIED CODE
                self.observations.append(flatten_dict_of_arrays(game_state.features))
                player1Action = self.player_one.take_turn(game_state)
                self.actions.append(value_to_discrete_4(player1Action["Move"].index))
                game_state = self.game.act(player1Action)
                
                # rewards from TensorRTS
                # self.rewards.append(game_state.reward)

                # rewards from custom function
                self.rewards.append(shaped_rewards(self.game.tensors,self.game.tensor_power,self.game.clusters))
                
                
                self.dones.append(game_state.done)
            except Exception as ex:
                raise Runtime_Failure_Exception(True, False, ex)
            
            if (self.game.is_game_over is False):
                if self.player_two is None: 
                    game_state = self.game.opponent_act()
                else:
                    game_state = GameRunner.flip_board(game_state)
                    try:
                        game_state = self.game.act(self.player_two.take_turn(game_state), False, True)

                        #MODIFIED CODE
                        #if the game ends and it was on player 2's move, we want to still mark that the game is done
                        if(game_state.done == True) :
                            self.dones[-1] = True

                    except Exception as ex: 
                        raise Runtime_Failure_Exception(False, True, ex)
                    
            num_game_turns += 1

        #who won
        p_one = self.game.tensor_power(0)
        p_two = self.game.tensor_power(1)

        self.results = GameResult(p_one, p_two)
        self.player_one.on_game_over(self.results.player_one_win(), self.results.tie())
        if self.player_two is not None:
            self.player_two.on_game_over(self.results.player_two_win(), self.results.tie())
        if self.enable_printouts :
            if self.results.player_one_win():
                print("The First Player won!")
            elif self.results.player_two_win():
                print("The Second Player won!")

def generate_dataset(num_runs,bot1,bot2):
    datasetDict = {
        'observations': [],
        'actions': [], 
        'rewards': [], 
        'dones': [],
    }

    #generates training data
    for _ in range(num_runs):
        game_runner = GameRunnerSaveStates(enable_printouts=False)

        observation = game_runner.game.observe()
        action_space = game_runner.game.action_space()

        player1 = bot1.create_instance(game_runner.game.observe(),game_runner.game.action_space())
        player2 = bot2.create_instance(game_runner.game.observe(),game_runner.game.action_space())

        game_runner.assign_players(player1, player2)
        game_runner.run(100)

        datasetDict['observations'].append(game_runner.observations)
        datasetDict['actions'].append(game_runner.actions)
        datasetDict['rewards'].append(game_runner.rewards)
        datasetDict['dones'].append(game_runner.dones)

    # print(game_runner.results.player_one_win())
    from datasets import Dataset
    dataset = Dataset.from_dict(datasetDict)
    return dataset
