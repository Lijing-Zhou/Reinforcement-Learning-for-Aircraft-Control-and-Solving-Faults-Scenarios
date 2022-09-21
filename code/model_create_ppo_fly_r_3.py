

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from multirotor.env_ppo_fly_r import SimGym
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
env = SimGym(Simulation(sim_frequency_hz=30.0,
                           aircraft=Aircraft(name='aircraft',
                                             body_mass=5.0, motor_mass=2.0,
                                             max_motor_thrust=140.0,
                                             boom_length=2.0, body_area=0.5),
                           init_conditions=np.zeros((4, 3)),
                           wind_present=False))

model = PPO('MlpPolicy', env, verbose=1,tensorboard_log="./ppo_fly_r_1_tensorboard/")
model.learn(total_timesteps=int(1e6))
model.save("ppo_fly_r_1") 
del model  # delete trained model to demonstrate loading


model = PPO('MlpPolicy', env, verbose=1,tensorboard_log="./ppo_fly_r_2_tensorboard/")
model.learn(total_timesteps=int(1e6))
model.save("ppo_fly_r_2") 
del model  # delete trained model to demonstrate loading

model = PPO('MlpPolicy', env, verbose=1,tensorboard_log="./ppo_fly_r_3_tensorboard/")
model.learn(total_timesteps=int(1e6))
model.save("ppo_fly_r_3") 
del model  # delete trained model to demonstrate loading

