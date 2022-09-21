
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from multirotor.env_ppo_remain_stable import SimGym
#from multirotor.env_ppo_fly_to_attitude_g import SimGym
#from multirotor.env_ppo_fly_left import SimGym
#from multirotor.env_ppo_fly_left_1 import SimGym
#from multirotor.env_ppo_fly_left_2 import SimGym
#from multirotor.env_ppo_fly_r import SimGym
#from multirotor.env_ppo_fly_r1 import SimGym
#from multirotor.env_ppo_fly_r2 import SimGym
from multirotor.simulator import Simulation
from multirotor.aircraft import Aircraft
import time as times
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def model_test1():
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

    model = PPO.load("ppo_remain_stable4", env=env)
    #model = PPO.load("ppo_fly_to_attitude_g", env=env)
    #model = PPO.load("ppo_fly_left", env=env)
    #model = PPO.load("ppo_fly_left1", env=env)
    #model = PPO.load("ppo_fly_left2", env=env)
    #model = PPO.load("ppo_fly_r", env=env)
    #model = PPO.load("ppo_fly_r1", env=env)
    #model = PPO.load("ppo_fly_r2", env=env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print("Mean reward: {}, std: {}".format(mean_reward, std_reward))
    obs = env.reset()

    for i in range(1000):

        action, _states = model.predict(obs, deterministic=True)  
        action_copy=action.copy()
        #actions.append(np.reshape(action_copy,1))
        #actions.append(np.reshape(action_copy,2))
        actions.append(np.reshape(action_copy,1))
                
        obs, rewards, dones, info = env.step(action)
        
        obs_copy=obs.copy()
        obss.append(np.reshape(obs_copy,12))

        rewards_copy=rewards
        rewards_s.append(np.reshape(rewards_copy,1))

        env.render()

    
    i_order=np.arange(1,1001)
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
    columns_pos = ['x', 'y', 'z']
    a1 = pd.DataFrame(data=states_df[['x','y','z']], index=i_order, columns=columns_pos)
    columns_ang = ['roll', 'pitch', 'yaw']
    a2 = pd.DataFrame(data=states_df[['roll', 'pitch', 'yaw']], index=i_order, columns=columns_ang)
    columns_lin = ['u', 'v', 'w']
    a3 = pd.DataFrame(data=states_df[['u', 'v', 'w',]], index=i_order, columns=columns_lin)
    columns_ang_v = ['p', 'q', 'r']
    a4 = pd.DataFrame(data=states_df[['p', 'q', 'r']], index=i_order, columns=columns_ang_v)
    a1= a1.rename(columns={'x':'x1','y':'y1','z':'z1' } )
    a3= a3.rename(columns={'u':'u1','v':'v1','w':'w1' } )
    return a1,a2,a3,a4
 



def model_test2():
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

    model = PPO.load("ppo_remain_stable5", env=env)
    #model = PPO.load("ppo_fly_to_attitude_g", env=env)
    #model = PPO.load("ppo_fly_left", env=env)
    #model = PPO.load("ppo_fly_left1", env=env)
    #model = PPO.load("ppo_fly_left2", env=env)
    #model = PPO.load("ppo_fly_r", env=env)
    #model = PPO.load("ppo_fly_r1", env=env)
    #model = PPO.load("ppo_fly_r2", env=env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print("Mean reward: {}, std: {}".format(mean_reward, std_reward))
    obs = env.reset()

    for i in range(1000):

        action, _states = model.predict(obs, deterministic=True)  
        action_copy=action.copy()
        #actions.append(np.reshape(action_copy,1))
        #actions.append(np.reshape(action_copy,2))
        actions.append(np.reshape(action_copy,1))
                
        obs, rewards, dones, info = env.step(action)
        
        obs_copy=obs.copy()
        obss.append(np.reshape(obs_copy,12))

        rewards_copy=rewards
        rewards_s.append(np.reshape(rewards_copy,1))

        env.render()

    
    i_order=np.arange(1,1001)
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
    columns_pos = ['x', 'y', 'z']
    a1 = pd.DataFrame(data=states_df[['x','y','z']], index=i_order, columns=columns_pos)
    columns_ang = ['roll', 'pitch', 'yaw']
    a2 = pd.DataFrame(data=states_df[['roll', 'pitch', 'yaw']], index=i_order, columns=columns_ang)
    columns_lin = ['u', 'v', 'w']
    a3 = pd.DataFrame(data=states_df[['u', 'v', 'w',]], index=i_order, columns=columns_lin)
    columns_ang_v = ['p', 'q', 'r']
    a4 = pd.DataFrame(data=states_df[['p', 'q', 'r']], index=i_order, columns=columns_ang_v)
    a1= a1.rename(columns={'x':'x2','y':'y2','z':'z2' } )
    a3= a3.rename(columns={'u':'u2','v':'v2','w':'w2' } )
    return a1,a2,a3,a4


def model_test3():
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

    model = PPO.load("ppo_remain_stable6", env=env)
    #model = PPO.load("ppo_fly_to_attitude_g", env=env)
    #model = PPO.load("ppo_fly_left", env=env)
    #model = PPO.load("ppo_fly_left1", env=env)
    #model = PPO.load("ppo_fly_left2", env=env)
    #model = PPO.load("ppo_fly_r", env=env)
    #model = PPO.load("ppo_fly_r1", env=env)
    #model = PPO.load("ppo_fly_r2", env=env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print("Mean reward: {}, std: {}".format(mean_reward, std_reward))
    obs = env.reset()

    for i in range(1000):

        action, _states = model.predict(obs, deterministic=True)  
        action_copy=action.copy()
        #actions.append(np.reshape(action_copy,1))
        #actions.append(np.reshape(action_copy,2))
        actions.append(np.reshape(action_copy,1))
                
        obs, rewards, dones, info = env.step(action)
        
        obs_copy=obs.copy()
        obss.append(np.reshape(obs_copy,12))

        rewards_copy=rewards
        rewards_s.append(np.reshape(rewards_copy,1))

        env.render()

    
    i_order=np.arange(1,1001)
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
    columns_pos = ['x', 'y', 'z']
    a1 = pd.DataFrame(data=states_df[['x','y','z']], index=i_order, columns=columns_pos)
    columns_ang = ['roll', 'pitch', 'yaw']
    a2 = pd.DataFrame(data=states_df[['roll', 'pitch', 'yaw']], index=i_order, columns=columns_ang)
    columns_lin = ['u', 'v', 'w']
    a3 = pd.DataFrame(data=states_df[['u', 'v', 'w',]], index=i_order, columns=columns_lin)
    columns_ang_v = ['p', 'q', 'r']
    a4 = pd.DataFrame(data=states_df[['p', 'q', 'r']], index=i_order, columns=columns_ang_v)
    a1= a1.rename(columns={'x':'x3','y':'y3','z':'z3' } )
    a3= a3.rename(columns={'u':'u3','v':'v3','w':'w3' } )
    return a1,a2,a3,a4

  

def plot_all(a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4):
    i_order=np.arange(1,1001)
    pos_df = pd.concat([a1,b1,c1],axis=1)
    columns_posz = ['z1', 'z2', 'z3']
    posz_df=pd.DataFrame(data=pos_df[['z1', 'z2', 'z3']], index=i_order, columns=columns_posz)
    posz = posz_df.plot()
    posz.set_xlabel("Time [ms]")
    posz.set_ylabel("Position [m]")
    plt.savefig('a1.eps', format='eps')
    
    ang_df = pd.concat([a2,b2,c2],axis=1)
    ang = ang_df.plot()
    ang.set_xlabel("Time [ms]")
    ang.set_ylabel("Angle [degree]")


    lin_df = pd.concat([a3,b3,c3],axis=1)
    lin = lin_df.plot()
    columns_linw = ['w1', 'w2', 'w3']
    linw_df=pd.DataFrame(data=lin_df[['w1', 'w2', 'w3']], index=i_order, columns=columns_linw)
    linw=linw_df.plot()
    linw.set_xlabel("Time [ms]")
    linw.set_ylabel("linear Velocity [m/s]")
    plt.savefig('a2.eps', format='eps')

    ang_v_df = pd.concat([a4,b4,c4],axis=1)
    av = ang_v_df.plot()
    av.set_xlabel("Time [ms]")
    av.set_ylabel("Angular Velocity [rad/s]")    


    '''
    columns_ang = ['roll', 'pitch', 'yaw']
    ang_df = pd.DataFrame(data=states_df[['roll', 'pitch', 'yaw']], index=i_order, columns=columns_ang)
    ang =  ang_df.plot()
    ang.set_xlabel("Time [ms]")
    ang.set_ylabel("Angle [degree]")

    columns_lin = ['u', 'v', 'w']
    lin_df = pd.DataFrame(data=states_df[['u', 'v', 'w',]], index=i_order, columns=columns_lin)
    lin = lin_df.plot()
    lin.set_xlabel("Time [ms]")
    lin.set_ylabel("linear Velocity [m/s]")

    columns_ang_v = ['p', 'q', 'r']
    ang_v_df = pd.DataFrame(data=states_df[['p', 'q', 'r']], index=i_order, columns=columns_ang_v)
    av = ang_v_df.plot()
    av.set_xlabel("Time [ms]")
    av.set_ylabel("Angular Velocity [rad/s]")    


    '''





    plt.show()
    return pos_df



  

if __name__ == '__main__':
    # change sim frequency and total simulation time here
    s11,s12,s13,s14=model_test1()
    s21,s22,s23,s24=model_test2()
    s31,s32,s33,s34=model_test3()
    pos=plot_all(s11,s12,s13,s14,s21,s22,s23,s24,s31,s32,s33,s34)