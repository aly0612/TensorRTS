#The packages accelerate and datasets are needed

import sys
sys.path.append("../..")  # Add the grandparent directory


from entity_gym.env import Observation
from typing import Dict, List, Mapping, Tuple, Set 
from entity_gym.env import *    

from training_scripts import shaped_rewards,flatten_dict_of_arrays,value_to_discrete_4,GameRunnerSaveStates,generate_dataset
from tournament_runner import Bot
from TensorRTS import Agent
import os

import DTscripts
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments


import random

class Random_Agent(Agent):
    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> None: 
        super().__init__(init_observation, action_space)

    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]:
        mapping = {}


        move = random.randrange(0, 4)
        # if current_game_state.features['Tensor'][1][2] > 0 :
        #     #rush
        #     mapping['Move'] = GlobalCategoricalAction(0, self.action_space['Move'].index_to_label[0])
        # else:
        #     #advance
        #     mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[2])
        mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[move])
        return mapping
    
    def on_game_start(self, is_player_one:bool, is_player_two:bool) -> None:
        return super().on_game_start(is_player_one, is_player_two)
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)

### Load in bots to use as training data

MODEL_SAVE_PATH = "./saved_models/random_test2"

RANDOM_AGENT_PATH = os.path.join( os.path.dirname(__file__), "trainingBots/randomBot")


player1 = Bot(bot_dir="/Users/den/Documents/CS/classwork/4900/gameAi-pa4/bots/randomBot/")

player2 = Bot(bot_dir="/Users/den/Documents/CS/classwork/4900/gameAi-pa4/bots/randomBot/")


#generate 1000 runs of random vs random : (max moves are capped at 100, can be changed within this function if needed)
rndmData = generate_dataset(1000,player1,player2)


collator = DTscripts.DecisionTransformerGymDataCollator(rndmData)
config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim)
model = DTscripts.TrainableDT(config)

os.environ["WANDB_DISABLED"] = "true" 
training_args = TrainingArguments(
    output_dir="output/",
    remove_unused_columns=False,
    num_train_epochs=200,
    per_device_train_batch_size=64,
    learning_rate=1e-4,
    weight_decay=1e-4,
    warmup_ratio=0.1,
    optim="adamw_torch",
    max_grad_norm=0.25,
    use_mps_device=True,
    # evaluation_strategy="steps",
    # eval_steps = 15 #needs a evaluation dataset 

)
training_args = training_args.set_save(strategy="steps", steps=500)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=rndmData,
    data_collator=collator,
)

trainer.train()
trainer.save_model(MODEL_SAVE_PATH)
print("model save under:",MODEL_SAVE_PATH)
