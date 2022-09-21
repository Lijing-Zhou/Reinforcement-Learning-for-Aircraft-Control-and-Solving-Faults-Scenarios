import numpy as np
import math
import random
from scipy.spatial.transform import Rotation as R


def add_simple_wind(lin_vel, windfield=np.array([2.0, 2.0, 0.0])):
    lin_vel = lin_vel + windfield
    return lin_vel


class WindModel:
    def __init__(self,
                 wind_speed,
                 ground_wind,
                 wind_dir,
                 sim_dt,
                 wingspan,
                 gust_model='3',
                 seed=42):
        self.wind_dir = wind_dir
        self.steady_wind_vec = np.array([wind_speed*math.sin(wind_dir), wind_speed*math.cos(wind_dir), 0.0])
        self.turb = TustinTurbModel(sim_dt, wingspan, ground_wind, wind_dir, gust_model, seed)
    
    def update_wind(self, height, airspeed):
        """
        Update wind with combined turbulance and gust model
        """
        lin_gust_vec, _ = self.turb.update_tustin_turb(height, airspeed)
        earth_gust_vec = self.turb.wind_to_earth(lin_gust_vec)
        return earth_gust_vec + self.steady_wind_vec


class TustinTurbModel:
    def __init__(self,
                 sim_dt,
                 wingspan,
                 ground_wind,
                 wind_dir,
                 gust_model='3',
                 seed=42
                 ):
        self.sim_dt = sim_dt
        self.wingspan = wingspan * 3.28
        self.ground_wind = ground_wind * 3.28
        self.wind_dir = wind_dir
        self.prob_index = gust_model
        self.gust_prob, self.gust_alt = self._create_gust_prob()
        random.seed(seed)

        self.xi_u_km1 = 0.0
        self.xi_v_km1 = 0.0
        self.xi_w_km1 = 0.0
        self.xi_p_km1 = 0.0
        self.xi_q_km1 = 0.0
        self.xi_r_km1 = 0.0

        self.nu_u_km1 = 0.0
        self.nu_v_km1 = 0.0
        self.nu_w_km1 = 0.0
        self.nu_p_km1 = 0.0
        
        self.xi_v_km2 = 0.0
        self.xi_w_km2 = 0.0
        self.nu_v_km2 = 0.0
        self.nu_v_km2 = 0.0
        self.nu_w_km2 = 0.0

    @staticmethod
    def _create_gust_prob():
        alt_index = (500.0, 1750.0, 3750.0, 7500.0, 15000.0, 25000.0, 35000.0,
                     45000.0, 55000.0, 65000.0, 75000.0, 80000.0)  # these are all in feet
        gust = {'1': (3.2, 2.2, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                '2': (4.2, 3.6, 3.3, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                '3': (6.6, 6.9, 7.4, 6.7, 4.6, 2.7, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0),
                '4': (8.6, 9.6, 10.6, 10.1, 8.0, 6.6, 5.0, 4.2, 2.7, 0.0, 0.0, 0.0),
                '5': (11.8, 13.0, 16.0, 15.1, 11.6, 9.7, 8.1, 8.2, 7.9, 4.9, 3.2, 2.1),
                '6': (15.6, 17.6, 23.0, 23.6, 22.1, 20.0, 16.0, 15.1, 12.1, 7.9, 6.2, 5.1),
                '7': (18.7, 21.5, 28.4, 30.2, 30.7, 31.0, 25.2, 23.1, 17.5, 10.7, 8.4, 7.2)}
        return gust, alt_index

    def get_gust_prob(self, prob_index, height):
        gust_vector = self.gust_prob[prob_index]
        gust_prob = np.interp(height, self.gust_alt, gust_vector)
        return gust_prob

    def update_tustin_turb(self, height, airspeed):
        # Calculate turb amplitude
        height = abs(height)
        if height == 0.0:
            height = 10.0
        if height <= 1000:
            L_u = height / math.pow((0.177 + 0.000823 * height), 1.2)
            L_w = height
            sig_w = 0.1 * self.ground_wind
            sig_u = sig_w / math.pow((0.177 + 0.000823 * height), 0.4)
        elif height <= 2000:
            L_u = L_w = 1000.0 + ((height - 1000.0) / 1000.0) * 750.0
            sig_u = sig_w = (0.1 * self.ground_wind) + (height - 1000.0)/1000.0 *\
                            (self.get_gust_prob(self.prob_index, height) - 0.1 * self.ground_wind)
        else:
            L_u = L_w = 1750.
            sig_u = sig_w = self.get_gust_prob(self.prob_index, height)

        sig_p = (1.9 / math.sqrt(L_w * self.wingspan)) * sig_w
        L_p = math.sqrt(L_w * self.wingspan) / 2.6
        tau_u = L_u / airspeed
        tau_w = L_w / airspeed
        tau_p = L_p / airspeed
        tau_q = (4.0 * self.wingspan / math.pi) / airspeed
        tau_r = (3.0 * self.wingspan / math.pi) / airspeed
        nu_u = random.uniform(-1., 1.)
        nu_v = random.uniform(-1., 1.)
        nu_w = random.uniform(-1., 1.)
        nu_p = random.uniform(-1., 1.)

        omega_w = airspeed / L_w
        omega_v = airspeed / L_u
        C_BL = (1/tau_u)/math.tan((self.sim_dt/2)/tau_u)
        C_BLp = (1/tau_p)/math.tan((self.sim_dt/2)/tau_p)
        C_BLq = (1/tau_q)/math.tan((self.sim_dt/2)/tau_q)
        C_BLr = (1/tau_r)/math.tan((self.sim_dt/2)/tau_r)

        xi_u = -(1 - C_BL*tau_u)/(1 + C_BL*tau_u)*self.xi_u_km1 \
               + sig_u*math.sqrt(2*tau_u/self.sim_dt)/(1 + C_BL*tau_u)*(nu_u + self.nu_u_km1)
        
        xi_v = -2*(math.pow(omega_v, 2)) - math.pow(C_BL, 2)/math.pow((omega_v + C_BL), 2)*self.xi_v_km1 \
               - math.pow((omega_v - C_BL), 2)/math.pow((omega_v + C_BL), 2) * self.xi_v_km2 \
               + sig_u*math.sqrt(3*omega_v/self.sim_dt)/math.pow((omega_v + C_BL), 2)*((C_BL + omega_v/math.sqrt(3.))*nu_v
               + 2/math.sqrt(3.)*omega_v*self.nu_v_km1
               + (omega_v/math.sqrt(3.) - C_BL)*self.nu_v_km2)

        xi_w = -2*(math.pow(omega_w, 2)) - math.pow((C_BL), 2)/math.pow((omega_w + C_BL), 2)*self.xi_w_km1 \
               - math.pow((omega_w - C_BL), 2)/math.pow((omega_w + C_BL), 2) * self.xi_w_km2 \
               + sig_w*math.sqrt(3*omega_w/self.sim_dt)/math.pow((omega_w + C_BL), 2)*((C_BL + omega_w/math.sqrt(3.))*nu_w \
               + 2/math.sqrt(3.)*omega_w*self.nu_w_km1 \
               + (omega_w/math.sqrt(3.) - C_BL)*self.nu_w_km2)
    
        xi_p = -(1 - C_BLp*tau_p)/(1 + C_BLp*tau_p)*self.xi_p_km1 + sig_p*math.sqrt(2*tau_p/self.sim_dt)/(1 + C_BLp*tau_p) * (nu_p + self.nu_p_km1)
        
        xi_q = -(1 - 4*self.wingspan*C_BLq/math.pi/airspeed)/(1 + 4*self.wingspan*C_BLq/math.pi/airspeed) * self.xi_q_km1 \
               + C_BLq/airspeed/(1 + 4*self.wingspan*C_BLq/math.pi/airspeed) * (xi_w - self.xi_w_km1)
    
        xi_r = -(1 - 3*self.wingspan*C_BLr/math.pi/airspeed)/(1 + 3*self.wingspan*C_BLr/math.pi/airspeed) * self.xi_r_km1 \
               + C_BLr/airspeed/(1 + 3*self.wingspan*C_BLr/math.pi/airspeed) * (xi_v - self.xi_v_km1)


        self.xi_v_km2 = self.xi_w_km1
        self.xi_w_km2 = self.xi_w_km1
        self.nu_v_km2 = self.nu_v_km1
        self.nu_w_km2 = self.nu_w_km1

        self.xi_u_km1 = xi_u
        self.xi_v_km1 = xi_v
        self.xi_w_km1 = xi_w
        self.xi_p_km1 = xi_p
        self.xi_q_km1 = xi_q
        self.xi_r_km1 = xi_r

        self.nu_u_km1 = nu_u
        self.nu_v_km1 = nu_v
        self.nu_w_km1 = nu_w
        self.nu_p_km1 = nu_p
        
        lin_gust_vel = np.array([xi_u, xi_v, xi_w])
        rot_gust_vel = np.array([xi_p, xi_q, xi_r])
        return lin_gust_vel/3.28, rot_gust_vel
    
    def wind_to_earth(self, gust_vec):
        euler_ang = np.array([0.0, 0.0, self.wind_dir])
        r = R.from_euler('xyz', euler_ang)
        return r.apply(gust_vec)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    turb = TustinTurbModel(1/30.0, 2.5, 5.0, 45.0)
    lin_vels = []
    rot_vels = []
    for idx in range(20000):
        height = 10000.0
        airspeed = 100.0
        lin_vel, rot_vel = turb.update_tustin_turb(height, airspeed)
        # gust_vel = turb.wind_to_earth(lin_vel)
        # gust_vel += [20.0, 20.0, 0.0]
        gust_vel = lin_vel
        lin_vels.append(gust_vel)
        rot_vels.append(rot_vel)

    plt.plot([idx for idx in range(20000)], lin_vels)
    
    plt.show()
    