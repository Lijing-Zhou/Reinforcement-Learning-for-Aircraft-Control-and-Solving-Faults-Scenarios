

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from multirotor.env_ppo_remain_stable import SimGym
from multirotor.simulator import Simulation
from multirotor.aircraft import Aircraft
import time as times
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





start_time = times.time()
actions=[]
rewards_s=[]
obss=[]
# Create environment
env = SimGym(Simulation(sim_frequency_hz=100.0,
                           aircraft=Aircraft(name='aircraft',
                                             body_mass=5.0, motor_mass=2.0,
                                             max_motor_thrust=140.0,
                                             boom_length=2.0, body_area=0.5),
                           init_conditions=np.zeros((4, 3)),
                           wind_present=False))

model = PPO('MlpPolicy', env, verbose=1,tensorboard_log="./ppo_remain_stable4_tensorboard/")
model.learn(total_timesteps=int(1e4))
model.save("ppo_remain_stable4") 
del model  
model = PPO('MlpPolicy', env, verbose=1,tensorboard_log="./ppo_remain_stable5_tensorboard/")
model.learn(total_timesteps=int(1e5))
model.save("ppo_remain_stable5") 
del model  
model = PPO('MlpPolicy', env, verbose=1,tensorboard_log="./ppo_remain_stable6_tensorboard/")
model.learn(total_timesteps=int(1e6))
model.save("ppo_remain_stable6") 
del model  
end_time = times.time()
print("耗时: {:.2f}秒".format(end_time - start_time))

