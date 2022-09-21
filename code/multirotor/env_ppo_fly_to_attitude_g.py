import numpy as np
import gym
from gym import spaces
from multirotor.simulator import Simulation
import math
import matplotlib.pyplot as plt

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

        self.action_space = spaces.Box(np.array([-1.0]),
                                       np.array([1.0]))

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
        mean=10
        sigma=2
        #50
        #10
        #1
        x=state[0,2]
        left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma))
        right = np.exp(-(x - mean)**2 / (2 * sigma))
        reward=left * right
        return reward

    def _is_done(self):
        return self._time > self.terminal_time

    def step(self, action):
        self.sim.set_controls(t0=action[0], t1=action[0], t2=action[0], t3=action[0])
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
    