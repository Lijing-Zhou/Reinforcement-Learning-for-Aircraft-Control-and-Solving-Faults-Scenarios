from tkinter import Y
import numpy as np
import gym
from gym import spaces
from multirotor.simulator import Simulation
import math
import matplotlib.pyplot as plt
import random


#ppo_fly_to_attitude_g
class SimGym(gym.Env):
    def __init__(self, simulation: Simulation):
        self.sim = simulation
        # TODO: shouldn't need this look at pendulum as this is used in mppi.py
        self._time = 0.0
        self._dt = self.sim.sim_dt
        state = self.sim.get_states().flatten()
        # state = np.append(state, self._dt)
        self.state = state
        self.terminal_time = 10.0

        self.action_space = spaces.Box(np.array([-1.0, -1.0,-1.0]),
                                       np.array([1.0, 1.0, 1.0]))

        #self.action_space = spaces.Box(np.array([-1.0, -1.0, -1.0,]),
        #                              np.array([1.0, 1.0, 1.0]))


        self.observation_space = spaces.Box(np.array([np.inf] * 12), np.array([-np.inf] * 12))

    def reset(self):
        self.sim.set_aircraft_state(np.zeros((4, 3)))
        state = self.sim.get_states()
        self.state = state.flatten()
        self._time = 0.0
        return self.state

    @staticmethod
    def _hover_reward(state):
        
        x=state[0,0]
        y=state[0,1]
        z=state[0,2]


        dis=(10-x)**2+(10-y)**2+(10-z)**2
        mean=0
        sigma=1
        left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma))
        right = np.exp(-(dis - mean)**2 / (2 * sigma))
        reward=left * right
        
        '''
        roll_angle=state[1,0]
        if roll_angle>2.5 or roll_angle<-2.5:
            reward=-1
        if roll_angle>3.1 or roll_angle<-3.1:
            reward=-2           
        '''
        '''
        roll_angle=state[1,0]
        mean1=0
        sigma1=1
        left1 = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma1))
        right1 = np.exp(-(roll_angle - mean1)**2 / (2 * sigma1))
        reward2=left1 * right1

        reward=2*reward1+reward2
        '''


        return reward

    '''
    def _is_done(self):
        return self._time > self.terminal_time
    '''

    def _is_done(self):

        done_factor=0
        if self._time > self.terminal_time:
            done_factor=1

        roll_angle=self.state[3]    
        if roll_angle>3.1 or roll_angle<-3.1:
            done_factor=1    
        pitch_angle=self.state[4]  
        if pitch_angle>3.1 or pitch_angle<-3.1:
            done_factor=1 
        yaw_angle=self.state[5]      
        if yaw_angle>3.1 or yaw_angle<-3.1:
            done_factor=1 
        



        return  done_factor

    def step(self, action):
        self.sim.set_controls(t0=0.227*math.sin(random.uniform(0,6.28)), t1=action[0], t2=action[1], t3=action[2])
        #self.sim.set_controls(t0=0.5*math.sin(time)+0.4*math.cos.(time), t1=action[1], t2=action[2], t3=action[3])
        state = self.sim.run()
        reward = self._hover_reward(state)  # set reward to 0 for now as not aiming to optimize
        self._time += self._dt
        # print(self._time)
        done = self._is_done()  # Non episodic env so no final state
        
        state = state.flatten()
        # state = np.append(state, self._time)
        self.state = state
        return state, reward, done, {}

    def render(self, mode='human'):
        return None


    def close(self):
        return None
    
'''
        mean_x=0
        sigma_x=1
        x=state[0,0]
        left_x = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma_x))
        right_x = np.exp(-(x - mean_x)**2 / (2 * sigma_x))
        reward_x=left_x * right_x

        mean_y=10
        sigma_y=4
        y=state[0,1]
        left_y = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma_y))
        right_y = np.exp(-(y - mean_y)**2 / (2 * sigma_y))
        reward_y=left_y * right_y
        
        mean_z=0
        sigma_z=1
        z=state[0,2]
        left_z = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma_z))
        right_z = np.exp(-(z - mean_z)**2 / (2 * sigma_z))
        reward_z=left_z * right_z

        reward= reward_x+ 100*reward_y+ reward_z

        return reward
     '''