

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from multirotor.env_ppo_fly_left_1 import SimGym
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
# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1,tensorboard_log="./ppo_fly_left1_tensorboard/")
#tensorboard_log="./ppo_SimGym_tensorboard/")
# Train the agent
model.learn(total_timesteps=int(1e6))
#le5====no jmodel--model=failed
# Save the agent
model.save("ppo_fly_left1") 
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = PPO.load("ppo_fly_left1", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print("Mean reward: {}, std: {}".format(mean_reward, std_reward))
end_time = times.time()

print("耗时: {:.2f}秒".format(end_time - start_time))

