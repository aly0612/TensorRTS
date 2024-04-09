from stable_baselines3 import A2C
import os
import time
import wandb
from TensorRTS_Env import TensorRTS_GymEnv
from stable_baselines3.common.callbacks import BaseCallback
from natsort import natsorted

class WandbLoggingCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.win_count = 0
        self.loss_count = 0
        self.game_count = 0

    def _on_step(self):
            infos = self.locals['infos']  # Extracts info for all environments
            for info in infos:
                if 'game_over' in info and info['game_over']:  # Check if a game has ended
                    self.game_count += 1
                    if info.get('win', False):  # get win key from info
                        self.win_count += 1
                    if info.get('lost', False):  # get lost key from info
                        self.loss_count += 1

        # Only log after each game to avoid too frequent logging
            if 'game_over' in infos[0] and infos[0]['game_over']:
                win_rate = self.win_count / self.game_count if self.game_count else 0
                loss_rate = self.loss_count / self.game_count if self.game_count else 0
                wandb.log({
                    'win_rate': win_rate, 
                    'loss_rate': loss_rate, 
                    'win_count': self.win_count, 
                    'loss_count': self.loss_count, 
                    'games_played': self.game_count
            })
            return True
        

# Initialize callback
callback = WandbLoggingCallback()

# Initialize wandb
run = wandb.init(

    project="TensorRTS",

    config={
         "architecture": "CNN",
        "learning_rate": 5e-4,
        "n_steps": 5,
        "algorithm": "A2C",
        "policy": "MultiInputPolicy",
        "timesteps": 1e6,
        "env": "TensorRTS_Env",
        "batch_size": 128,
        "n_epochs": 10,
        "gae_lambda": 0.98,
        "ent_coef": 0.001,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
  #  sync_tensorboard=True,
)

model_root_dir = 'models'
log_root_dir = 'logs'

current_time = str(int(time.time()))

models_dir = os.path.join(model_root_dir, current_time)
log_dir = os.path.join(log_root_dir, current_time)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = TensorRTS_GymEnv()

model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

# Use natural sort to find the latest model.
list_folders = natsorted(os.listdir(model_root_dir), reverse=True)
for folder in list_folders:
    list_models = natsorted(os.listdir(os.path.join(model_root_dir, folder)), reverse=True)
    if len(list_models) > 0:
        model_name = os.path.join(model_root_dir, folder, list_models[0])
        print('Loading model from', model_name)
        model = A2C.load(model_name, env=env, tensorboard_log=log_dir)
        break

TIMESTEPS = 1e6
iters = 0



model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C", callback=callback)


model.save(os.path.join(models_dir, str(iters)))

env.close()

print(f"Saved model to {models_dir}/{iters}")

# [Optional] Finish the wandb run
wandb.finish()