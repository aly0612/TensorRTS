# https://enn-trainer.readthedocs.io/en/latest/quick-start-guide.html
#
# First, run: python train.py --config=config.ron --checkpoint-dir=checkpoints

from enn_trainer import load_checkpoint, RogueNetAgent
from entity_gym.env import *
checkpoint = load_checkpoint('checkpoints/latest-step000000020480')
agent = RogueNetAgent(checkpoint.state.agent)
obs = Observation(
   global_features=[0, 0],
   features={
       "Trap": [[-5, 0], [-2, 0], [0, 3], [0, -4], [0, -3]],
       "Treasure": [[2, 0]],
   },
   done=True,
   reward=0.0,
   actions={"move": GlobalCategoricalActionMask()},
)
action, predicted_return = agent.act(obs)
