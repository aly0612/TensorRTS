# See tutorials at https://gymnasium.farama.org/

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List, Mapping, Tuple, Set

class TensorRTS_Env(gym.Env):
    """
    Custom Environment for TensorRTS
    """
    metadata = {'render.modes': ['console']}
    
    def __init__(
        self,
        mapsize: int = 32,
        nclusters: int = 6,
        maxdots: int = 9,
        enable_prinouts : bool = True,
        attack_speed: int = 2,
        e_to_m: float = 0.2,
        boom_factor: float = 0.2,
        attack_adv: float = 0.8
    ):
        super(TensorRTS_Env, self).__init__()
    
        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0, high=maxdots, shape=(mapsize,)),
            "tensors": spaces.Box(low=0, high=255, shape=(2,4))  
                # First dimension: 2: tensor 1/2; 
                # Second dimension: 4: position, dimension of the tensor, x, y
        })

        self.enable_printouts = enable_prinouts
        
        if self.enable_printouts:
            print(f"LinearRTS sb3 -- Mapsize: {mapsize}")

        self.mapsize = mapsize
        self.maxdots = maxdots
        self.nclusters = nclusters
        self.clusters: List[List[int]] = []  # The inner list has a size of 2 (position, number of dots).
        self.tensors:  List[List[int]] = []  # The inner list has a size of 4 (position, dimension, x, y).
        self.attack_speed = attack_speed
        self.e_to_m = e_to_m
        self.boom_factor = boom_factor
        self.attack_adv = attack_adv

    def _get_observation(self):
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
 
        position = random.randint(0, self.mapsize // 4)
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
        
        if action == 0: #advance
            self.tensors
            
                
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

from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":  # This is to train

    env = TensorRTS_Env()
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
