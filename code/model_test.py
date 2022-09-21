

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
#from multirotor.env_ppo_remain_stable import SimGym
#from multirotor.env_ppo_fly_to_attitude_g import SimGym
#from multirotor.env_ppo_fly_left import SimGym
#from multirotor.env_ppo_fly_left_1 import SimGym
#from multirotor.env_ppo_fly_left_2 import SimGym
#from multirotor.env_ppo_fly_r import SimGym
#from multirotor.env_ppo_fly_r1 import SimGym
from multirotor.env_ppo_fly_r2 import SimGym
from multirotor.simulator import Simulation
from multirotor.aircraft import Aircraft
import time as times
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def model_test():
    actions=[]
    rewards_s=[]
    obss=[]
    #100 hz
    env = SimGym(Simulation(sim_frequency_hz=100.0,
                           aircraft=Aircraft(name='aircraft',
                                             body_mass=5.0, motor_mass=2.0,
                                             max_motor_thrust=140.0,
                                             boom_length=2.0, body_area=0.5),
                           init_conditions=np.zeros((4, 3)),
                           wind_present=False))     
    #ppo_fly_to_attitude_g
    #model = PPO.load("ppo_remain_stable1", env=env)
    #model = PPO.load("ppo_fly_to_attitude_g", env=env)
    #model = PPO.load("ppo_fly_left", env=env)
    #model = PPO.load("ppo_fly_left1", env=env)
    #model = PPO.load("ppo_fly_left2", env=env)
    #model = PPO.load("ppo_fly_r", env=env)
    #model = PPO.load("ppo_fly_r1", env=env)
    model = PPO.load("ppo_fly_r2", env=env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print("Mean reward: {}, std: {}".format(mean_reward, std_reward))
    obs = env.reset()

    for i in range(10000):

        action, _states = model.predict(obs, deterministic=True)  
        action_copy=action.copy()
        #actions.append(np.reshape(action_copy,1))
        #actions.append(np.reshape(action_copy,2))
        actions.append(np.reshape(action_copy,3))
                
        obs, rewards, dones, info = env.step(action)
        
        obs_copy=obs.copy()
        obss.append(np.reshape(obs_copy,12))

        rewards_copy=rewards
        rewards_s.append(np.reshape(rewards_copy,1))

        env.render()

    
    i_order=np.arange(1,10001)
    columns = ['x', 'y', 'z',
                'roll', 'pitch', 'yaw', 
                'u', 'v', 'w',
                'p', 'q', 'r']
    
    states_df = pd.DataFrame(data=obss, index=i_order, columns=columns)
    #states_df['z']=-states_df['z']
    #states_df['w']=-states_df['w']
    #plot all states
    states_df['roll']= np.rad2deg(states_df['roll'])
    states_df['pitch']= np.rad2deg(states_df['pitch'])
    states_df['yaw']= np.rad2deg(states_df['yaw'])
    '''   
    ax = states_df.plot()
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Value")
    # plot invidiual state eg height
    '''

    columns_pos = ['x', 'y', 'z']
    pos_df = pd.DataFrame(data=states_df[['x','y','z']], index=i_order, columns=columns_pos)
    pos = pos_df.plot()
    pos.set_xlabel("Time [ms]")
    pos.set_ylabel("Position [m]")
    plt.savefig('1.eps', format='eps')

    #control_states_df = control_states_df.set_index('time')

    columns_ang = ['roll', 'pitch', 'yaw']
    ang_df = pd.DataFrame(data=states_df[['roll', 'pitch', 'yaw']], index=i_order, columns=columns_ang)
    ang =  ang_df.plot()
    ang.set_xlabel("Time [ms]")
    ang.set_ylabel("Angle [degree]")
    plt.savefig('2.eps', format='eps')


    columns_lin = ['u', 'v', 'w']
    lin_df = pd.DataFrame(data=states_df[['u', 'v', 'w',]], index=i_order, columns=columns_lin)
    lin = lin_df.plot()
    lin.set_xlabel("Time [ms]")
    lin.set_ylabel("linear Velocity [m/s]")
    plt.savefig('3.eps', format='eps')

    columns_ang_v = ['p', 'q', 'r']
    ang_v_df = pd.DataFrame(data=states_df[['p', 'q', 'r']], index=i_order, columns=columns_ang_v)
    av = ang_v_df.plot()
    av.set_xlabel("Time [ms]")
    av.set_ylabel("Angular Velocity [rad/s]")    
    plt.savefig('4.eps', format='eps')



    


    plt.show()
    return states_df,actions,rewards_s, obs,action,rewards

if __name__ == '__main__':
    # change sim frequency and total simulation time here
    states_df,actions,rewards_s, obs,action,rewards=model_test()