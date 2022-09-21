import numpy as np
import gym
from gym import spaces
from multirotor.simulator import Simulation


class SimGym(gym.Env):
    def __init__(self, simulation: Simulation):
        self.sim = simulation
        # TODO: shouldn't need this look at pendulum as this is used in mppi.py
        self._time = 0.0
        state = self.sim.get_states().flatten()
        self.state = state
        self.action_space = spaces.Box(np.array([-1.0, -1.0, -1.0, -1.0]),
                                       np.array([1.0, 1.0, 1.0, 1.0]))
        self.observation_space = spaces.Box(np.array([np.inf] * 12), np.array([-np.inf] * 12))

    def reset(self):
        self.sim.set_aircraft_state(np.zeros((4, 3)))
        state = self.sim.get_states()
        self.state = state.flatten()
        return self.state

    @staticmethod
    def _hover_reward(state):
        state_weights = np.zeros((4, 3))
        state_weights[0, :] = 1.0
        reference = np.zeros((4, 3))
        reward = np.sum(state_weights * np.abs(state - reference))
        return reward

    def step(self, action):
        self.sim.set_controls(t0=action[0], t1=action[1], t2=action[2], t3=action[3])
        state = self.sim.run()
        reward = self._hover_reward(state)  # set reward to 0 for now as not aiming to optimize
        done = False  # Non episodic env so no final state
        state = state.flatten()
        self.state = state
        return state, reward, done, {}

    def close(self):
        return None
    