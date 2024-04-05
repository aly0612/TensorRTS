```
(enn_py38) [ohu0417@pitzer-login01 treasure_hunt]$ python train.py --config=config.ron --checkpoint-dir=checkpoints
 2048/100000 | meanrew 1.76e-02 | explained_var  0.00 | entropy  1.38 | episodic_reward 2.39e-01 | episode_length 1.28e+01 | episodes 46 | entities 10.7 | fps 71| eps 767
 4096/100000 | meanrew 1.12e-02 | explained_var  0.07 | entropy  1.37 | episodic_reward 4.64e-01 | episode_length 2.70e+01 | episodes 28 | entities 10.5 | fps 91| eps 966
 6144/100000 | meanrew 1.07e-02 | explained_var  0.14 | entropy  1.36 | episodic_reward 5.71e-01 | episode_length 3.38e+01 | episodes 28 | entities 10.4 | fps 86| eps 892
 8192/100000 | meanrew 1.51e-02 | explained_var  0.11 | entropy  1.35 | episodic_reward 8.00e-01 | episode_length 6.19e+01 | episodes 30 | entities 10.3 | fps 95| eps 977
10240/100000 | meanrew 1.61e-02 | explained_var  0.20 | entropy  1.34 | episodic_reward 6.67e-01 | episode_length 4.29e+01 | episodes 27 | entities 10.2 | fps 4| eps 43
12288/100000 | meanrew 1.27e-02 | explained_var  0.25 | entropy  1.32 | episodic_reward 7.50e-01 | episode_length 6.27e+01 | episodes 32 | entities 10.0 | fps 5| eps 50
14336/100000 | meanrew 1.86e-02 | explained_var  0.35 | entropy  1.33 | episodic_reward 1.18e+00 | episode_length 6.87e+01 | episodes 28 | entities 10.0 | fps 6| eps 61
16384/100000 | meanrew 1.27e-02 | explained_var  0.36 | entropy  1.32 | episodic_reward 6.83e-01 | episode_length 5.34e+01 | episodes 41 | entities 9.9 | fps 4| eps 49
18432/100000 | meanrew 2.34e-02 | explained_var  0.36 | entropy  1.31 | episodic_reward 1.29e+00 | episode_length 6.70e+01 | episodes 31 | entities 9.9 | fps 5| eps 51
20480/100000 | meanrew 2.78e-02 | explained_var  0.35 | entropy  1.31 | episodic_reward 1.00e+00 | episode_length 4.60e+01 | episodes 41 | entities 9.7 | fps 4| eps 47
22528/100000 | meanrew 3.03e-02 | explained_var  0.37 | entropy  1.29 | episodic_reward 1.09e+00 | episode_length 6.40e+01 | episodes 33 | entities 9.4 | fps 4| eps 45
```
![Screenshot 2024-02-14 at 7 33 41 PM](https://github.com/drchangliu/RL4SE/assets/1515931/74a57535-fa87-4d6b-ac10-3b29731ea80c)


```
(enn_py38) [ohu0417@pitzer-login04 treasure_hunt]$ python
Python 3.8.18 (default, Sep 11 2023, 13:40:15) 
[GCC 11.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> exec(open("load_checkpoint.py").read())
>>> action
{'move': GlobalCategoricalAction(index=1, label='down', probs=array([0.15364452, 0.27247274, 0.04293703, 0.53094566], dtype=float32))}
>>> predicted_return
0.5157977342605591
>>> obs.features["Trap"] = [[-4, 0]]
>>> action, predicted_return = agent.act(obs)
>>> action
{'move': GlobalCategoricalAction(index=3, label='right', probs=array([0.1552546 , 0.26702395, 0.04343911, 0.5342824 ], dtype=float32))}
>>> predicted_return
1.3330386877059937
>>> from entity_gym.runner import CliRunner
>>> from entity_gym.examples.tutorial import TreasureHunt
>>> CliRunner(TreasureHunt(), agent).run()
Environment: TreasureHunt
Global features: x_pos, y_pos
Entity Trap: x_pos, y_pos
Entity Treasure: x_pos, y_pos
Categorical move: up, down, left, right

Step 0
Reward: 0.0
Total: 0.0
Predicted return: 5.120e-01
Global features: x_pos=0, y_pos=0
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=0, y_pos=-2)
7 Treasure(x_pos=-5, y_pos=2)
8 Treasure(x_pos=2, y_pos=0)
9 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 20.9% 1/down 25.6% 2/left 7.9% 3/right 45.6%)
Step 1
Reward: 0.0
Total: 0.0
Predicted return: 5.259e-01
Global features: x_pos=0, y_pos=-1
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=0, y_pos=-2)
7 Treasure(x_pos=-5, y_pos=2)
8 Treasure(x_pos=2, y_pos=0)
9 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 30.9% 1/down 12.2% 2/left 5.2% 3/right 51.7%)
Step 2
Reward: 0.0
Total: 0.0
Predicted return: 5.120e-01
Global features: x_pos=0, y_pos=0
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=0, y_pos=-2)
7 Treasure(x_pos=-5, y_pos=2)
8 Treasure(x_pos=2, y_pos=0)
9 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 20.9% 1/down 25.6% 2/left 7.9% 3/right 45.6%)
Step 3
Reward: 0.0
Total: 0.0
Predicted return: 5.259e-01
Global features: x_pos=0, y_pos=-1
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=0, y_pos=-2)
7 Treasure(x_pos=-5, y_pos=2)
8 Treasure(x_pos=2, y_pos=0)
9 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 30.9% 1/down 12.2% 2/left 5.2% 3/right 51.7%)
Step 4
Reward: 0.0
Total: 0.0
Predicted return: 6.345e-01
Global features: x_pos=1, y_pos=-1
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=0, y_pos=-2)
7 Treasure(x_pos=-5, y_pos=2)
8 Treasure(x_pos=2, y_pos=0)
9 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 36.4% 1/down 6.7% 2/left 3.9% 3/right 52.9%)

Step 5
Reward: 0.0
Total: 0.0
Predicted return: 6.956e-01
Global features: x_pos=2, y_pos=-1
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=0, y_pos=-2)
7 Treasure(x_pos=-5, y_pos=2)
8 Treasure(x_pos=2, y_pos=0)
9 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 46.6% 1/down 4.7% 2/left 6.2% 3/right 42.5%)
Step 6
Reward: 1.0
Total: 1.0
Predicted return: 2.750e-01
Global features: x_pos=2, y_pos=0
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=0, y_pos=-2)
7 Treasure(x_pos=-5, y_pos=2)
8 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 30.7% 1/down 17.2% 2/left 26.6% 3/right 25.5%)
Step 7
Reward: 0.0
Total: 1.0
Predicted return: 2.606e-01
Global features: x_pos=1, y_pos=0
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=0, y_pos=-2)
7 Treasure(x_pos=-5, y_pos=2)
8 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 32.1% 1/down 17.8% 2/left 17.3% 3/right 32.8%)
Step 8
Reward: 0.0
Total: 1.0
Predicted return: 2.303e-01
Global features: x_pos=1, y_pos=-1
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=0, y_pos=-2)
7 Treasure(x_pos=-5, y_pos=2)
8 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 29.2% 1/down 20.5% 2/left 20.1% 3/right 30.2%)
Step 9
Reward: 0.0
Total: 1.0
Predicted return: 1.943e-01
Global features: x_pos=0, y_pos=-1
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=0, y_pos=-2)
7 Treasure(x_pos=-5, y_pos=2)
8 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 25.4% 1/down 25.0% 2/left 14.5% 3/right 35.1%)
Step 10
Reward: 1.0
Total: 2.0
Predicted return: -2.922e-02
Global features: x_pos=0, y_pos=-2
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=-5, y_pos=2)
7 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 37.1% 1/down 14.3% 2/left 15.1% 3/right 33.4%)
Step 11
Reward: 0.0
Total: 2.0
Predicted return: -1.065e-01
Global features: x_pos=0, y_pos=-3
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=-5, y_pos=2)
7 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 37.4% 1/down 14.0% 2/left 15.0% 3/right 33.7%)
Step 12
Reward: 0.0
Total: 2.0
Predicted return: -2.920e-02
Global features: x_pos=0, y_pos=-2
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=-5, y_pos=2)
7 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 37.1% 1/down 14.3% 2/left 15.1% 3/right 33.4%)
Step 13
Reward: 0.0
Total: 2.0
Predicted return: -1.065e-01
Global features: x_pos=0, y_pos=-3
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=-5, y_pos=2)
7 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 37.4% 1/down 14.0% 2/left 15.0% 3/right 33.7%)
Step 14
Reward: 0.0
Total: 2.0
Predicted return: -1.032e-01
Global features: x_pos=1, y_pos=-3
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=-5, y_pos=2)
7 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 35.9% 1/down 14.9% 2/left 19.7% 3/right 29.5%)
Step 15
Reward: 0.0
Total: 2.0
Predicted return: -1.917e-02
Global features: x_pos=1, y_pos=-2
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=-5, y_pos=2)
7 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 36.5% 1/down 14.4% 2/left 20.3% 3/right 28.8%)
Step 16
Reward: 0.0
Total: 2.0
Predicted return: -2.918e-02
Global features: x_pos=0, y_pos=-2
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=-5, y_pos=2)
7 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 37.1% 1/down 14.3% 2/left 15.1% 3/right 33.4%)
Step 17
Reward: 0.0
Total: 2.0
Predicted return: 3.369e-02
Global features: x_pos=0, y_pos=-1
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=-5, y_pos=2)
7 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 36.7% 1/down 14.7% 2/left 15.9% 3/right 32.7%)
Step 18
Reward: 0.0
Total: 2.0
Predicted return: 5.907e-02
Global features: x_pos=1, y_pos=-1
Entities
0 Trap(x_pos=2, y_pos=1)
1 Trap(x_pos=-1, y_pos=0)
2 Trap(x_pos=4, y_pos=4)
3 Trap(x_pos=-2, y_pos=-5)
4 Trap(x_pos=4, y_pos=2)
5 Treasure(x_pos=-5, y_pos=-5)
6 Treasure(x_pos=-5, y_pos=2)
7 Treasure(x_pos=1, y_pos=3)
Choose move (0/up 38.3% 1/down 12.9% 2/left 20.8% 3/right 28.0%)
```
![Screenshot 2024-02-14 at 7 58 07 PM](https://github.com/drchangliu/RL4SE/assets/1515931/322428f5-4938-4de0-a0da-4a1a79ac4896)


