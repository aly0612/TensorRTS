from entity_gym.env import *
from enn_trainer import TrainConfig, State, init_train_state, train

import sys
sys.path.append("..")     # Add the parent directory
sys.path.append("../..")  # Add the grandparent directory
sys.path.append("../../..")  

from TensorRTS import TensorRTS

import hyperstate
@hyperstate.stateful_command(TrainConfig, State, init_train_state)
def main(state_manager: hyperstate.StateManager) -> None:
  train(state_manager=state_manager, env=TensorRTS)

if __name__ == "__main__":
  main()