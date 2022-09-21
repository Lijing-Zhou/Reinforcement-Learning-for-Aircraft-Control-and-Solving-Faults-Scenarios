from scipy.spatial.transform import Rotation as R
import math
import numpy as np


class Aircraft:
    
    def __init__(self,
                 name,
                 body_mass,
                 motor_mass,
                 max_motor_thrust,
                 boom_length,
                 body_area,
                 motor_lag=False,
                 dt=30.0,
                 lag_size=0.1,
                 state=np.zeros((4, 3)),
                 control_state=np.zeros((4, 1)),
                 ):
        self.name = name
        self.body_mass = body_mass
        self.motor_mass = motor_mass
        self.mass = body_mass + (4 * motor_mass)
        self.max_motor_thrust = max_motor_thrust
        self.boom_length = boom_length
        self.body_area = body_area
        self.motor_pos = self.calculate_motor_pos()
        self.inertia = self.calculate_inertia()
        self.lag_comp = self.calc_motor_lag(lag_size, dt)
        self.motor_lag = motor_lag
        self.motor_comm = np.zeros((4, 1))
        self.state = state
        self.control_state = control_state

    def get_state(self):
        return self.state

    def get_control_state(self):
        return self.control_state
    
    def set_control_state(self, t0=0.0, t1=0.0, t2=0.0, t3=0.0):
        self.control_state[0] = t0
        self.control_state[1] = t1
        self.control_state[2] = t2
        self.control_state[3] = t3
        return self.control_state

    def set_state(self, state):
        self.state = state

    #TODO: Find a way to speed this up, from_euler is slow
    def body_to_earth(self, vec):
        r = R.from_euler('xyz', self.state[1, :])
        return r.apply(vec)

    def calculate_motor_pos(self):
        boom_pos = self.boom_length * math.sin(math.radians(45))
        b0 = [boom_pos, boom_pos, 0.0]  # front-right
        b1 = [-boom_pos, boom_pos, 0.0]  # back-right
        b2 = [-boom_pos, -boom_pos, 0.0]  # back-left
        b3 = [boom_pos, -boom_pos, 0.0]  # front-left
        return np.array([b0, b1, b2, b3])

    def calculate_inertia(self):
        inertia = np.zeros([3, 3])
        for pos in self.motor_pos:
            inertia_diag = self.motor_mass * np.square(pos)
            inertia[0, 0] += inertia_diag[0]
            inertia[1, 1] += inertia_diag[1]
            inertia[2, 2] += inertia_diag[2]
        inertia[2, 2] = 16 
        return inertia

    @staticmethod
    def calc_motor_lag(lag_size, dt):
        return dt / (lag_size + dt)

    def calculate_motor_force(self):
        t_vec = np.array([0, 0, -1])

        # Add lag component if desired
        if self.motor_lag:
            self.motor_comm = ((1-self.lag_comp) * self.motor_comm) + (self.lag_comp * self.control_state)
        else:
            self.motor_comm = self.control_state

        # Calculate the force from motors in ground plane
        t = self.motor_comm * t_vec * self.max_motor_thrust
        force_body = np.sum(t, axis=0)
        force = self.body_to_earth(force_body)

        # Calculate the torque from motors in ground plane
        tor = np.cross(self.motor_pos, t)
        torque_body = np.sum(tor, axis=0)
        torque = self.body_to_earth(torque_body)
        return force, torque

    def calculate_drag(self):
        BODY_DRAG_COEFFICIENT = 1.28
        airspeed = self.state[2, :]
        if np.linalg.norm(airspeed) == 0.0:
            return np.array([0.0, 0.0, 0.0])
        unit_direction = airspeed / np.linalg.norm(airspeed)
        force = - 0.5 * 1.225 * (np.linalg.norm(airspeed) ** 2) * self.body_area * BODY_DRAG_COEFFICIENT*unit_direction
        return force

    def calculate_weight(self):
        force = self.mass * 9.81 * np.array([0, 0, 1])
        return force

    @staticmethod
    def calculate_const_wind_force():
        force_dir = np.array([1, 0, 0])
        force_size = 30.0
        force = force_size * force_dir
        return force

    def calculate_total_force(self):
        motor_force, motor_torque = self.calculate_motor_force()
        drag_force = self.calculate_drag()
        weight_force = self.calculate_weight()
        tot_force = motor_force + drag_force + weight_force
        tot_torque = motor_torque
        return tot_force, tot_torque


# default = Aircraft(name='default', body_mass=5.0, motor_mass=2.0, max_motor_thrust=35.0,
#                    boom_length=2.0, body_area=0.5)
