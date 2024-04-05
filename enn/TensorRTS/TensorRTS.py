import random
import abc
import os
import json
import argparse

import numpy as np
import gymnasium as gym

from typing import Dict, List, Mapping, Tuple, Set
from gymnasium import spaces
from entity_gym.env import *
from entity_gym.runner import CliRunner

from stable_baselines3.common.env_checker import check_env
class Runtime_Failure_Exception(Exception): 
    def __init__(self, responsible_bot_is_player_one : bool, responsible_bot_is_player_two : bool, parent_exception : Exception, *args: object) -> None:
        self.responsible_bot_is_player_one = responsible_bot_is_player_one
        self.responsible_bot_is_player_two = responsible_bot_is_player_two
        self.parent_exception = parent_exception
        super().__init__(*args)

class Turn_History(): 
    def __init__(self, board_features_observation : dict) -> None: 
        self.start_board_layout : dict = board_features_observation
        self.act_player_one : str = None
        self.act_player_two : str = None

    def set_player_one_action(self, action_player_one : str) -> None: 
        self.act_player_one = action_player_one

    def set_player_two_action(self, action_player_two : str) -> None: 
        self.act_player_two = action_player_two

class Tensor_Base(metaclass=abc.ABCMeta): 
    def __init__(self,
        mapsize: int = 32,
        nclusters: int = 6,
        ntensors: int = 2,  # There could be more than 2 players in the future
        maxdots: int = 9,
        enable_prinouts : bool = True,
        attack_speed: int = 2,
        e_to_m: float = 0.2,
        boom_factor: float = 0.2,
        attack_adv: float = 0.8): 

        self.enable_printouts = enable_prinouts
        
        if self.enable_printouts:
            print(f"LinearRTS -- Mapsize: {mapsize}")

        self.mapsize = mapsize
        self.maxdots = maxdots
        self.nclusters = nclusters
        self.clusters: List[List[int]] = []  # The inner list has a size of 2 (position, number of dots).
        self.tensors:  List[List[int]] = []  # The inner list has a size of 4 (position, dimension, x, y).
        # Adjustable parameters for game balance. Default values are as given in PA4 on GitHub
        self.attack_speed = attack_speed
        self.e_to_m = e_to_m
        self.boom_factor = boom_factor
        self.attack_adv = attack_adv

        self.turn_record : list[Turn_History] = []
        self.current_turn : Turn_History = None

    def record_turn(self): 
        assert(self.current_turn is not None)
        self.turn_record.append(self.current_turn)
        self.current_turn = None

    @abc.abstractmethod
    def reset(self): 
        pass

    @abc.abstractmethod
    def observe(self): 
        pass

class TensorRTS_GymEnv(gym.Env, Tensor_Base):

    def __init__(self, mapsize: int = 32, nclusters: int = 6, ntensors: int = 2, maxdots: int = 9, enable_prinouts: bool = True, attack_speed: int = 2, e_to_m: float = 0.2, boom_factor: float = 0.2, attack_adv: float = 0.8):
        super().__init__(mapsize, nclusters, ntensors, maxdots, enable_prinouts, attack_speed, e_to_m, boom_factor, attack_adv)
        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0, high=maxdots, shape=(mapsize,)),
            "tensors": spaces.Box(low=0, high=255, shape=(2,4))  
                # First dimension: 2: tensor 1/2; 
                # Second dimension: 4: position, dimension of the tensor, x, y
        })

    def observe(self): 
        # Generate the observation based on self.clusters and self.tensors
        map_values = np.zeros(self.mapsize, dtype=np.float32)

        for cluster in self.clusters:
            map_values[cluster[0]] = cluster[1]

        # Create observation
        # Flatten the observation components into a 1D array
        # Ensure map_values has the correct shape (assuming mapsize is defined elsewhere)
        if map_values.shape != (self.mapsize,):
            raise ValueError("map_values must have shape (mapsize,)")

        # Create the observation dictionary
        observation = {
            "map": map_values.copy(),  # Copy to avoid unintended modification
            "tensors": self.tensors.copy()  # Copy to avoid unintended modification
        }
        # Assuming mapsize and maxdots are defined elsewhere

        # Create empty map with appropriate shape
        map_values = np.zeros(self.mapsize, dtype=np.float32)

        # Create tensors array with example data (modify as needed)
        tensors = np.array([[1, 3, 100, 50],  # Tensor 1, dimension 3, at (100, 50)
                    [2, 2, 20, 30]], dtype=np.float32)  # Tensor 2, dimension 2, at (20, 30)

        # Combine map and tensors into observation dictionary
        observation = {
            "map": map_values.copy(),  # Copy to avoid unintended modification
            "tensors": tensors.copy()  # Copy to avoid unintended modification
        }
        return observation
    
    def _get_observation(self):
        return self.observe()
    
    def _get_info(self):
        information = {
            'mapsize': self.mapsize,
            'maxdots': self.maxdots,
            'nclusters': self.nclusters
        }
        return information

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Reset the state of the environment to an initial state
        positions = set()
        while len(positions) < self.nclusters // 2:
            position, b = random.choice(
                [[position, b] for position in range(self.mapsize // 2) for b in range(1, self.maxdots)]
            )
            if position not in positions:
                positions.add(position)
                self.clusters.append([position, b])
                self.clusters.append([self.mapsize - position - 1, b])
        self.clusters.sort()
 
        position = random.randint(0, self.mapsize // 2)
        self.tensors = [[position, 1, 2, 0], [self.mapsize - position - 1, 1, 2, 0]]
        # Starting positions are added for TP calculation
        self.starts = (self.tensors[0][0], self.tensors[1][0])

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # This game will not be truncated so it is always False
        # Return value: observation, reward, terminated, truncated, info. The last item "info" can be a dictionary containing additional information from the environment, such as debugging data or specific metrics.
        print(f"{action}")
        reward = 0
        terminated = False
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def render(self, mode='console'):
        returnString = " "
        divider = "\n---"

        for cluster in self.clusters:
            returnString += " | " + str(cluster)
            divider += "----"
        divider += "\n"

        returnString += divider
        print(returnString)

    def close(self):
        # Clean up when closing the environment
        pass

class TensorRTS(Environment, Tensor_Base):
    """
Simple TensorRTS on a linear map, the first epoch of TensorRTS, is intended to be the simplest RTS game.
    """

    def __init__(
        self,
        mapsize: int = 32,
        nclusters: int = 6,
        ntensors: int = 2,  # There could be more than 2 players in the future
        maxdots: int = 9,
        enable_prinouts : bool = True,
        attack_speed: int = 2,
        e_to_m: float = 0.2,
        boom_factor: float = 0.2,
        attack_adv: float = 0.8
    ):
        super().__init__(mapsize, nclusters, ntensors, maxdots, enable_prinouts, attack_speed, e_to_m, boom_factor, attack_adv)

    def set_state(self, state : Observation, initial_tensor_positions):
        self.done = state.done
        self.reward = state.reward
        self.clusters = state.features["Cluster"]
        self.tensors = state.features["Tensor"]
        self.starts = initial_tensor_positions

    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            entities={
                "Cluster": Entity(features=["position", "dot"]),
                "Tensor": Entity(features=["position", "dimension", "x", "y"]),
            }
        )

    def action_space(cls) -> Dict[ActionName, ActionSpace]:
        return {
            "Move": GlobalCategoricalActionSpace(
                ["advance", "retreat", "rush", "boom"],
            ),
        }

    def reset(self) -> Observation:
        positions = set()
        while len(positions) < self.nclusters // 2:
            position, b = random.choice(
                [[position, b] for position in range(self.mapsize // 2) for b in range(1, self.maxdots)]
            )
            if position not in positions:
                positions.add(position)
                self.clusters.append([position, b])
                self.clusters.append([self.mapsize - position - 1, b])
        self.clusters.sort()
 
        position = random.randint(0, self.mapsize // 4)  
            # 1/4 makes sure the left tensor is always in the left quarter, and the right tensor is always in the right quarter.
            # They will never start very close to each other, in which case the rusher will leave no chance to boomers.
        self.tensors = [[position, 1, 2, 0], [self.mapsize - position - 1, 1, 2, 0]]
        # Starting positions are added for TP calculation
        self.starts = (self.tensors[0][0], self.tensors[1][0])

        if self.enable_printouts:
            self.print_universe()

        return self.observe()
    
    def tensor_power(self, tensor_index) -> float :
        # A better tensor power calculation may be possible that doesn't depend heavily on whether the unit starts on the left or right
        if tensor_index == 0:
            f = self.tensors[tensor_index][3] * (1 + (self.tensors[tensor_index][0]-(self.starts[1]-self.starts[0])/2)/self.mapsize*self.attack_adv)
        else:
            f = self.tensors[tensor_index][3] * (1 + ((self.starts[1]-self.starts[0])/2-self.tensors[tensor_index][0])/self.mapsize*self.attack_adv)
        
        if self.enable_printouts:
            print(f"TP({tensor_index})=TP({self.tensors[tensor_index]})={f}")
        return f

    def observe(self) -> Observation:
        done = self.tensors[0][0] >= self.tensors[1][0]
        if done:
            reward = 10 if self.tensor_power(0) > self.tensor_power(1) else 0 if self.tensor_power(0) == self.tensor_power(1) else -10
        else:
            reward = 1.0 if self.tensors[0][1] > self.tensors[1][1] else 0.0
        return Observation(
            entities={
                "Cluster": (
                    self.clusters,
                    [("Cluster", i) for i in range(len(self.clusters))],
                ),
                "Tensor": (
                    self.tensors,
                    [("Tensor", i) for i in range(len(self.tensors))],
                ),
            },
            actions={
                "Move": GlobalCategoricalActionMask(),
            },
            done=done,
            reward=reward,
        )

    def act(self, actions: Mapping[ActionName, Action], trigger_default_opponent_action : bool = True, is_player_two : bool = False) -> Observation:
        action = actions["Move"]
        assert isinstance(action, GlobalCategoricalAction)
        
        if self.current_turn is None: 
            self.current_turn = Turn_History(self.observe().features)

        if not is_player_two: 
            self.current_turn.set_player_one_action(action.label)
        else:
            self.current_turn.set_player_two_action(action.label)
            self.record_turn()

        player_tensor = self.tensors[0]
        if is_player_two:
            player_tensor = self.tensors[1]
            
        if action.label == "advance":
            for _ in range(self.attack_speed):
                # ensure that the player can't move past the edge of the map
                if player_tensor[0] < self.mapsize - 1 and not is_player_two:
                    player_tensor[0] += 1
                    player_tensor[2] += self.collect_dots(player_tensor[0])
                elif player_tensor[0] > 0 and is_player_two:
                    player_tensor[0] -= 1
                    player_tensor[2] += self.collect_dots(player_tensor[0])
        elif action.label == "retreat":
            # ensure no negative movement
            if player_tensor[0] > 0 and not is_player_two:
                player_tensor[0] -= 1
                player_tensor[2] += self.collect_dots(player_tensor[0])
            elif player_tensor[0] < self.mapsize - 1 and is_player_two:
                player_tensor[0] += 1
                player_tensor[2] += self.collect_dots(player_tensor[0])
        elif action.label == "boom":
            if int(self.boom_factor * player_tensor[2] > 1):
                player_tensor[2] += int(self.boom_factor * player_tensor[2])
            else:
                player_tensor[2] += 1
        elif action.label == "rush":
            if player_tensor[2] >= 1:
                if int(self.e_to_m * player_tensor[2]) > 1:
                    player_tensor[1] = 2
                    player_tensor[3] += int(self.e_to_m * player_tensor[2])
                    player_tensor[2] -= int(self.e_to_m * player_tensor[2])
                else:
                    player_tensor[1] = 2 # the number of dimensions is now 2
                    player_tensor[2] -= 1
                    player_tensor[3] += 1

        if trigger_default_opponent_action:
            self.opponent_act()
        
        if self.enable_printouts:
            self.print_universe()

        return self.observe()

    def opponent_act(self):         # This is the rush AI.
        if self.tensors[1][2]>0 :   # Rush if possile
            if int(self.e_to_m * self.tensors[1][2]) > 1:
                self.tensors[1][1] = 2
                self.tensors[1][2] -= int(self.e_to_m * self.tensors[1][2])
                self.tensors[1][3] += int(self.e_to_m * self.tensors[1][2])
            else:
                self.tensors[1][2] -= 1
                self.tensors[1][3] += 1
                self.tensors[1][1] = 2      # the number of dimensions is now 2
        else:                       # Otherwise Advance.
            for _ in range(self.attack_speed):
                if self.tensors[1][0] > 0:
                    self.tensors[1][0] -= 1
                    self.tensors[1][2] += self.collect_dots(self.tensors[1][0])

        return self.observe()

    def collect_dots(self, position):
        low, high = 0, len(self.clusters) - 1

        while low <= high:
            mid = (low + high) // 2
            current_value = self.clusters[mid][0]

            if current_value == position:
                dots = self.clusters[mid][1]
                self.clusters[mid][1] = 0
                return dots
            elif current_value < position:
                low = mid + 1
            else:
                high = mid - 1

        return 0        

    def print_universe(self):
        for j in range(self.mapsize):
            print(f" {j%10}", end="")
        print(" #")
        position_init = 0
        for i in range(len(self.clusters)):
            for j in range(position_init, self.clusters[i][0]):
                print("  ", end="")
            print(f" {self.clusters[i][1]}", end="")
            position_init = self.clusters[i][0]+1
        for j in range(position_init, self.mapsize):
            print("  ", end="")
        print(" ##")

        position_init = 0
        for i in range(len(self.tensors)):
            for j in range(position_init, self.tensors[i][0]):
                print("  ", end="")
            print(f"{self.tensors[i][2]}", end="")
            if self.tensors[i][3]>=0:
                print(f"-{self.tensors[i][3]}", end="")
            position_init = self.tensors[i][0]+1
        for j in range(position_init, self.mapsize):
            print("  ", end="")
        print(" ##")

class Interactive_TensorRTS(TensorRTS): 
    '''
    This class is a way of avoiding potential conflicts with enn_trainer so that we would not have to really touch 
    the core game class for anything required to "run" a game.
    '''
    def __init__(self,
        mapsize: int = 32,
        nclusters: int = 6,
        ntensors: int = 2,
        maxdots: int = 9, 
        enable_printouts : bool = True,
        trace_file: str = ""):   # No game trace is saved if the file name is "". 

        self.trace_file = trace_file 
        self.is_game_over = False

        super().__init__(mapsize, nclusters, ntensors, maxdots, enable_prinouts=enable_printouts)
            
    def reset(self): 
        super().reset()
        
        if self.trace_file:  # Check if file name is not empty
            with open(self.trace_file, "a") as file:
                file.write(f"{self.clusters}\n")
                file.write(f"{self.tensors}\n")

    def act(self, actions: Mapping[ActionName, Action],  trigger_default_opponent_action : bool = True, is_player_two : bool = False, print_universe : bool = False) -> Observation:
        obs_result = super().act(actions, False, is_player_two)

        if self.trace_file :  # Check if file name is not empty
            with open(self.trace_file, "a") as file:
                # file.write(f"{is_player_two} - {actions}\n")
                file.write(f"{self.clusters}\n")
                file.write(f"{self.tensors}\n")
            base_name, extension = os.path.splitext(self.trace_file)
            with open(base_name + "_actions" + extension, "a") as file:
                file.write(f"{is_player_two} - {actions}\n")
            
       
        if (obs_result.done == True):
            self.is_game_over = True

        return obs_result

class Agent(metaclass=abc.ABCMeta):
    def __init__(self, initial_observation : Observation, action_space : Dict[ActionName, ActionSpace], script_dir : str):
        self.previous_game_state = initial_observation
        self.action_space = action_space

    @abc.abstractmethod
    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]: 
        """Pure virtual function in which an agent should return the move that they will make on this turn.

        Returns:
            str: name of the action that will be taken
        """
        pass

    @abc.abstractmethod
    def on_game_start(self, is_player_one : bool, is_player_two : bool) -> None: 
        """Function which is called for the agent before the game begins.

        Args:
            is_player_one (bool): Set to true if the agent is playing as player one
            is_player_two (bool): Set to true if the agent is playing as player two
        """
        assert(is_player_one == True or is_player_two == True)

        self.is_player_one = is_player_one
        self.is_player_two = is_player_two

    @abc.abstractmethod
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        """Function which is called for the agent once the game is over.

        Args:
            did_i_win (bool): set to True if this agent won the game.
        """
        pass

class Random_Agent(Agent):
    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> None: 
        super().__init__(init_observation, action_space, script_dir="")

    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]:
        mapping = {}

        action_choice = random.randrange(0, 2)
        if (action_choice == 1): 
            mapping['Move'] = GlobalCategoricalAction(0, self.action_space['Move'].index_to_label[0])
        else: 
            mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[1])
        
        return mapping
    
    def on_game_start(self, is_player_one : bool, is_player_two : bool) -> None:
        return super().on_game_start(is_player_one, is_player_two)
     
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)

class GameResult():
    def __init__(self, board_size : int, max_step_count : int, player_one_win_count : int, player_two_win_count : int, is_create_exception : bool = False, additional_information : str = None, turns = None, player_one_name : str = None, player_two_name : str = None):
        self.board_size : int = board_size
        self.max_step_count : int = max_step_count
        self.player_one_win_count = player_one_win_count
        self.player_two_win_count = player_two_win_count
        self.is_create_exception = is_create_exception
        self.player_one_name = player_one_name
        self.player_two_name = player_two_name
        self.turns : list[Turn_History] = turns
        self.additional_information = additional_information

    def player_one_win(self) -> bool: 
        return self.player_one_win_count > self.player_two_win_count
    
    def player_two_win(self) -> bool: 
        return self.player_two_win_count > self.player_one_win_count
    
    def tie(self) -> bool: 
        return self.player_one_win_count == self.player_two_win_count

class GameRunner(): 
    def __init__(self, environment = None, enable_printouts : bool = False, trace_file : str = ""):
        self.enable_printouts = enable_printouts
        self.game = Interactive_TensorRTS(enable_printouts=enable_printouts, trace_file=trace_file)
        self.game.reset()

        self.player_one = None
        self.player_one_name = None
        self.player_two = None
        self.player_two_name = None
        self.trace_file = trace_file
        self.results : GameResult = None

    def get_turn_history(self) -> List[Turn_History]: 
        return self.game.turn_record
        
        # json_file_name = self.trace_file.split('.txt')[0] + ".json"
        # print(json_file_name)
        # with open(json_file_name, 'w+') as json_file: 
        #     json_file.write(serialized)
    
    def assign_players(self, first_agent : Agent, second_agent : Agent = None, first_agent_student_name : str = None, second_agent_student_name : str = None):
        self.player_one = first_agent

        self.player_one_name = first_agent_student_name
        self.player_two_name = second_agent_student_name

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
                game_state = self.game.act(self.player_one.take_turn(game_state))
            except Exception as ex:
                raise Runtime_Failure_Exception(True, False, ex)
            
            if (self.game.is_game_over is False):
                if self.player_two is None: 
                    game_state = self.game.opponent_act()
                else:
                    game_state = GameRunner.flip_board(game_state)
                    try:
                        game_state = self.game.act(self.player_two.take_turn(game_state), False, True)
                    except Exception as ex: 
                        raise Runtime_Failure_Exception(False, True, ex)
                    
            num_game_turns += 1

        #who won
        p_one = self.game.tensor_power(0)
        p_two = self.game.tensor_power(1)

        self.results = GameResult(self.game.mapsize, max_game_turns, p_one, p_two, turns=self.game.turn_record, player_one_name=self.player_one_name, player_two_name=self.player_two_name)
        self.player_one.on_game_over(self.results.player_one_win(), self.results.tie())
        if self.player_two is not None:
            self.player_two.on_game_over(self.results.player_two_win(), self.results.tie())

        print('---Match Info---')
        if (self.player_one_name is not None and self.player_two_name is not None): 
            print('Player names:')
            print(f'Player One: {self.player_one_name}')
            print(f'Player Two: {self.player_two_name}')

        if self.results.player_one_win():
            print("The First Player won!")
        elif self.results.player_two_win():
            print("The Second Player won!")
        print(f"Scores: {p_one} - {p_two}")


if __name__ == "__main__":  # This is to run wth a random agent
    parser = argparse.ArgumentParser(
        prog='TensorRTS',
        description='A simple RTS game'
    )
    parser.add_argument('-m', '--mode')
    args = parser.parse_args()

    mode = None
    if args.mode is None: 
        print("### Playing a game session with a random agent ... ")
        runner = GameRunner(enable_printouts=True, trace_file="GameTrace.txt")
        init_observation = runner.get_game_observation(is_player_two= False)
        random_agent = Random_Agent(init_observation, runner.game.action_space())

        runner.assign_players(random_agent)
        runner.run(max_game_turns = 300)
    elif args.mode == 'cli': 
        print("")
        print("### Playing a game session with console input ... ")
        env = TensorRTS()
        # The `CliRunner` can run any environment with a command line interface.
        CliRunner(env).run()
    else:
        env = TensorRTS_GymEnv()
        # This will check the custom environment and output warnings if any are found
        check_env(env)

        episodes = 50
        for i in range(episodes):
            done = False
            obs = env.reset()
            while not done:
                env.render()
                random_action = env.action_space.sample()
                print('action:', random_action)
                obs, reward, terminated, truncated, info = env.step(random_action)
                done = terminated or truncated
                print('reward:', reward, '\n')