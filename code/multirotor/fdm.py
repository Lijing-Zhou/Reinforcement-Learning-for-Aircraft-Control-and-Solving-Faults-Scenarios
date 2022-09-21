import numpy as np
import math
# import numba as nb
from multirotor.wind import WindModel


# @nb.jit(nopython=True)
def _calculate_linear_accel(force, mass):
    return force / mass


# @nb.jit(nopython=True)
def _calculate_angular_accel(torque, inertia):
    inv_inertia = np.linalg.pinv(inertia)
    return np.dot(inv_inertia, torque)


# @nb.jit(nopython=True)
def _euler_integrate(init_val, derivative_val, sim_dt):
    return init_val + (sim_dt * derivative_val)


class FDM:
    def __init__(self,
                 sim_dt,
                 aircraft,
                 init_conditions=np.zeros((4, 3)),
                 wind_present=False):
        self.aircraft = aircraft
        self.aircraft.state = init_conditions
        self.time = 0.0
        self.sim_dt = sim_dt
        self.wind_present = wind_present
        if wind_present:
            self.wind_model = WindModel(wind_speed=2.0, 
                                        ground_wind=1.0,
                                        wind_dir=45.0,
                                        sim_dt=sim_dt,
                                        wingspan=2.0)
    
    def reset(self, init_conditions, time=0.0):
        self.time = time
        self.aircraft.state = init_conditions

    def update(self):
        state = self.aircraft.state
        aircraft_force, aircraft_torque = self.aircraft.calculate_total_force()

        lin_accel = self.calculate_linear_accel(aircraft_force)
        ang_accel = self.calculate_angular_accel(aircraft_torque)

        # Use euler integration to update velocity and position values of state
        lin_vel = self.euler_integrate(self._get_lin_vel(state), lin_accel)
        ang_vel = self.euler_integrate(self._get_ang_vel(state), ang_accel)
        state = self._set_lin_vel(lin_vel, state)
        state = self._set_ang_vel(ang_vel, state)

        if self.wind_present:
            # Addition of Tustin windspeed model, designed for use with aircraft
            airspeed = np.linalg.norm(lin_vel)
            if airspeed <= 0.5:
                airspeed=0.5
            height = state[0, 2] - 500  # start above the horizon
            wind_lin_vel = self.wind_model.update_wind(height=height, airspeed=airspeed)
            lin_vel += wind_lin_vel

        pos = self.euler_integrate(self._get_pos(state), lin_vel)
        att = self.euler_integrate(self._get_att(state), ang_vel)
        att = self._bound_att(att)
        state = self._set_pos(pos, state)
        state = self._set_att(att, state)
        
        self.time += self.sim_dt
        self.aircraft.state = state
        return state

    def calculate_linear_accel(self, force):
        return _calculate_linear_accel(force, self.aircraft.mass)

    def calculate_angular_accel(self, torque):
        return _calculate_angular_accel(torque, self.aircraft.inertia)

    def euler_integrate(self, init_val, derivative_val):
        return _euler_integrate(init_val, derivative_val, self.sim_dt)

    def _bound_att(self, att):
        for idx, angle in enumerate(att):
            if angle < -math.pi:
                angle += 2 * math.pi
                att[idx] = angle
            elif angle > math.pi:
                angle -= 2 * math.pi
                att[idx] = angle
        return att


    @staticmethod
    def _get_pos(state):
        return state[0, :]

    @staticmethod
    def _get_att(state):
        return state[1, :]

    @staticmethod
    def _get_lin_vel(state):
        return state[2, :]

    @staticmethod
    def _get_ang_vel(state):
        return state[3, :]

    @staticmethod
    def _set_pos(pos, state):
        state[0, :] = pos
        return state

    @staticmethod
    def _set_att(att, state):
        state[1, :] = att
        return state

    @staticmethod
    def _set_lin_vel(lin_vel, state):
        state[2, :] = lin_vel
        return state
    
    @staticmethod
    def _set_ang_vel(ang_vel, state):
        state[3, :] = ang_vel
        return state
