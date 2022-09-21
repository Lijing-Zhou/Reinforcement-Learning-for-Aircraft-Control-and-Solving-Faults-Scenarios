'''
for key in list(globals().keys()):
 if (not key.startswith("_")) and (key !="key"):
     globals().pop(key)
del(key)     

'''


from multirotor.simulator import Simulation
from multirotor.aircraft import Aircraft
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as times


def power_up(sim_frequency_hz=100.0, sim_time=10.0):
    sim = Simulation(sim_frequency_hz=sim_frequency_hz,
                     aircraft=Aircraft(name='default', body_mass=5.0,
                                       motor_mass=2.0, max_motor_thrust=140.0,
                                       boom_length=2.0, body_area=0.5, motor_lag=True, lag_size=5),
                                       #body_area=0.5
                     init_conditions=np.zeros((4, 3)),
                     wind_present=False)
    arr_len = int(sim_time * sim_frequency_hz)
    time = np.empty(arr_len)
    states = []
    control_states = []
    force_drone_motors=[]
    torque_drone_motors=[]
    torque_drone_motors=[]
    force_drags=[]
    tot_forces=[]
    tot_torques=[]
    linear_accels=[]
    angular_accels=[]
    start_time = times.time()

    for idx in range(arr_len):
        state = sim.run()
        state_copy = state.copy()
        states.append(np.reshape(state_copy, 12))

        control_state = sim.get_control_states()
        control_state_copy = control_state.copy()
        control_states.append(np.reshape(control_state_copy, 4))
        
        current_time = sim.get_time()
        time[idx] = current_time
        # if current_time > start_time:

        force_drone_motor,torque_drone_motor=sim.aircraft.calculate_motor_force()
        force_drone_motor_copy=force_drone_motor.copy()
        force_drone_motors.append(force_drone_motor_copy)
        torque_drone_motor_copy=torque_drone_motor.copy()
        torque_drone_motors.append(torque_drone_motor_copy)

        force_drag=sim.aircraft.calculate_drag()
        force_drag_copy=force_drag.copy()
        force_drags.append(force_drag_copy)

        tot_force, tot_torque=sim.aircraft.calculate_total_force()
        tot_force_copy=tot_force.copy()
        tot_forces.append(tot_force_copy)
        tot_torque_copy=tot_torque.copy()
        tot_torques.append(tot_torque_copy)

        linear_accel=sim.fdm.calculate_linear_accel(tot_force)
        linear_accel_copy=linear_accel.copy()
        linear_accels.append(linear_accel_copy)


        angular_accel=sim.fdm.calculate_angular_accel(tot_torque)
        angular_accel_copy=angular_accel.copy()
        angular_accels.append(angular_accel_copy)



        #set controls between 0 and 1 at each timestep, in this case constant, play around with motor torques to see effects on aircraft states
        #sim.set_controls(t0=0.227, t1=0.227, t2=0.227, t3=0.227)
        #sim.set_controls(t0=0.22773217, t1=0.22773217, t2=0.22773217, t3=0.22773217)
        #sim.set_controls(t0=0.5, t1=0.5, t2=0.5, t3=0.5)
        
        #sim.set_controls(t0=0.3, t1=0.14, t2=0.5, t3=0.1)
        sim.set_controls(t0=0, t1=0, t2=0, t3=0)
        #sim.set_controls(t0=0.227, t1=-0.227, t2=0.227, t3=-0.227)
        #sim.set_controls(t0=-0.1, t1=-0.05, t2=-0.1, t3=-0.05)
        # sim.set_controls(t0=0.2646702528, t1=-0.1081023216, t2=-0.5589153171, t3=-0.1081023216)
        # sim.set_controls(t0=-0.227+0.1, t1=-0.227+0.1, t2=-0.227-0.1, t3=-0.227-0.1)
        #sim.set_controls(t0=0.22773217+0.1, t1=0.22773217+0.1, t2=0.22773217-0.1, t3=0.22773217-0.1)

       
        inertia=sim.aircraft.calculate_inertia()




    
    columns = ['x', 'y', 'z',
               'roll', 'pitch', 'yaw',
               'u', 'v', 'w',
               'p', 'q', 'r']
    states_df = pd.DataFrame(data=states, index=time, columns=columns)

    control_states_df = pd.DataFrame(control_states)
    control_states_df['time'] = time

    columns_force_drone_motors= ['Fx', 'Fy', 'Fz']
    force_drone_motors_df=pd.DataFrame(data=force_drone_motors, index=time, columns=columns_force_drone_motors)
    columns_torque_drone_motors= ['Ix', 'Iy', 'Iz']   
    torque_drone_motors_df=pd.DataFrame(data=torque_drone_motors, index=time, columns=columns_torque_drone_motors)
    columns_force_drags= ['Fx', 'Fy', 'Fz']   
    force_drags_df=pd.DataFrame(data=force_drags, index=time, columns=columns_force_drags) 


    columns_tot_forces= ['Fx', 'Fy', 'Fz']
    tot_forces_df=pd.DataFrame(data=tot_forces, index=time, columns=columns_tot_forces)
    columns_tot_torques= ['Ix', 'Iy', 'Iz']   
    tot_torques_df=pd.DataFrame(data= tot_torques, index=time, columns=columns_tot_torques)

    columns_linear_accels= ['ax', 'ay', 'az']
    linear_accels_df=pd.DataFrame(data=linear_accels, index=time, columns=columns_linear_accels)
    columns_angular_accels= ['a_roll', 'a_pitch', 'a_yaw']   
    angular_accels_df=pd.DataFrame(data=angular_accels, index=time, columns=columns_angular_accels)

    columns_pos = ['x', 'y', 'z']
    pos_df = pd.DataFrame(data=states_df[['x','y','z']], index=time, columns=columns_pos)

    columns_ang = ['roll', 'pitch', 'yaw']
    ang_df = pd.DataFrame(data=states_df[['roll', 'pitch', 'yaw']], index=time, columns=columns_ang)

    columns_lin = ['u', 'v', 'w']
    lin_df = pd.DataFrame(data=states_df[['u', 'v', 'w',]], index=time, columns=columns_lin)

    columns_ang_v = ['p', 'q', 'r']
    ang_v_df = pd.DataFrame(data=states_df[['p', 'q', 'r']], index=time, columns=columns_ang_v)

    #plot all states
    ax = states_df.plot()
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    


    # plot invidiual state eg height
    states_df.plot(y='z')
    states_df.plot(y='roll')
    states_df.plot(y='pitch')      
    states_df.plot(y='yaw')

    '''
    states_df.plot(y='x')
    states_df.plot(y='y')
    states_df.plot(y='z')

    states_df.plot(y='roll')
    states_df.plot(y='pitch')      
    states_df.plot(y='yaw')

    states_df.plot(y='u')
    states_df.plot(y='v')
    states_df.plot(y='w')

    states_df.plot(y='p')
    states_df.plot(y='q')
    states_df.plot(y='r')
    '''
    pos_df.plot() 
    ang_df.plot()   
    lin_df.plot() 
    ang_v_df.plot()   
    

    # states_df.plot(y='y')
    # states_df.plot(y='w')
    control_states_df = control_states_df.set_index('time')
    # control_states_df.plot()
    plt.show()

    end_time = times.time()
    print("耗时: {:.2f}秒".format(end_time - start_time))
    return states_df,control_states_df,inertia,force_drone_motors_df,torque_drone_motors_df,\
        force_drags_df,tot_forces_df,tot_torques_df,linear_accels_df,angular_accels_df


if __name__ == '__main__':
    # change sim frequency and total simulation time here
    states_df,control_states_df,inertia,force_drone_motors_df,torque_drone_motors_df,force_drags_df\
    ,tot_forces_df,tot_torques_df,linear_accels_df,angular_accels_df\
    =power_up(sim_frequency_hz=30.0, sim_time=10.0)
