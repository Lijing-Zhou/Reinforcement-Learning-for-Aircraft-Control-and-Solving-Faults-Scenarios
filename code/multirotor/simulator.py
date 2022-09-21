import os
import time
import numpy as np
from typing import Dict, Union
from multirotor.aircraft import Aircraft
from multirotor.fdm import FDM


class Simulation:

    def __init__(self,
                 sim_frequency_hz=60.0,
                 aircraft=Aircraft(name='default', body_mass=5.0, motor_mass=2.0, max_motor_thrust=35.0,
                                   boom_length=2.0, body_area=0.5),
                 init_conditions=np.zeros((4, 3)),
                 wind_present=False):
        self.sim_dt = 1.0 / sim_frequency_hz
        self.aircraft = aircraft
        self.fdm = FDM(self.sim_dt, self.aircraft, init_conditions, wind_present)

    def run(self):
        return self.fdm.update()

    def reset(self, conditions=np.zeros((4, 3)), start_time=0.0):
        return self.fdm.reset(init_conditions=conditions, time=start_time)

    def get_aircraft(self):
        return self.aircraft

    def get_aircraft_state(self):
        return self.aircraft.state
        
    def get_time(self):
        return self.fdm.time
    
    def get_states(self):
        return self.aircraft.state
    
    def get_control_states(self):
        return self.aircraft.control_state

    def set_controls(self, t0=0.0, t1=0.0, t2=0.0, t3=0.0):
        self.aircraft.set_control_state(t0=t0, t1=t1, t2=t2, t3=t3)
        return self.aircraft.control_state

    def set_aircraft_state(self, conditions):
        self.aircraft.set_state(conditions)
        return self.aircraft.control_state
