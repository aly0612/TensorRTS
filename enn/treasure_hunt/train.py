# https://enn-trainer.readthedocs.io/en/latest/quick-start-guide.html

from enn_trainer import TrainConfig, State, init_train_state, train
from entity_gym.examples.tutorial import TreasureHunt
import hyperstate

@hyperstate.stateful_command(TrainConfig, State, init_train_state)
def main(state_manager: hyperstate.StateManager) -> None:
    train(state_manager=state_manager, env=TreasureHunt)

if __name__ == "__main__":
    main()
