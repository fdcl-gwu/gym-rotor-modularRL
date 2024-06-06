"""
Reference: https://github.com/fdcl-gwu/uav_simulator/blob/main/scripts/trajectory.py
"""
import numpy as np
import datetime

import args_parse
from gym_rotor.envs.quad_utils import *

class TrajectoryGeneration:
    def __init__(self, env):
        # Hyperparameters:
        parser = args_parse.create_parser()
        args = parser.parse_args()

        """----------------------------------------------------------
            self.mode == 0 or self.mode == 1:  # idle and warm-up
        -------------------------------------------------------------
            self.mode == 2:  # take-off
        -------------------------------------------------------------
            self.mode == 3:  # landing
        -------------------------------------------------------------
            self.mode == 4:  # stay (hovering)
        -------------------------------------------------------------
            self.mode == 5:  # circle
        -------------------------------------------------------------
            self.mode == 6:  #  eight-shaped curve
        ----------------------------------------------------------"""
        self.mode = 0
        self.is_mode_changed = False
        self.is_landed = False
        self.e1 = env.e1

        self.is_realtime = False # if False, it is sim_time
        self.t0 = datetime.datetime.now()
        self.t = 0.0
        self.t_traj = 0.0
        self.dt = env.dt

        self.x_lim, self.v_lim, self.W_lim = env.x_lim, env.v_lim, env.W_lim

        self.x_init, self.v_init = np.zeros(3), np.zeros(3)
        self.R_init, self.W_init = np.identity(3), np.zeros(3)
        self.b1_init = np.zeros(3)
        self.theta_init = 0.0

        self.x_norm, self.v_norm, self.W_norm = np.zeros(3), np.zeros(3), np.zeros(3)
        self.x, self.v, self.W = np.zeros(3), np.zeros(3), np.zeros(3)
        self.R = np.identity(3)

        self.xd, self.vd, self.Wd = np.zeros(3), np.zeros(3), np.zeros(3)
        self.xd_norm, self.vd_norm, self.Wd_norm = np.zeros(3), np.zeros(3), np.zeros(3)
        self.b1d = np.array([1.,0.,0.]) # desired heading direction
        self.b3d = np.array([0.,0.,1.])

        # Integral terms:
        self.use_integral = True
        self.sat_sigma = 1.
        self.eIX = IntegralErrorVec3() # Position integral error
        self.eIR = IntegralError() # Attitude integral error
        self.eIX.set_zero() # Set all integrals to zero
        self.eIR.set_zero()
        self.alpha, self.beta = args.alpha, args.beta # addressing noise or delay
        self.eIx_lim, self.eIb1_lim = env.eIx_lim, env.eIb1_lim

        # Geometric tracking controller:
        self.xd_2dot, self.xd_3dot, self.xd_4dot = np.zeros(3), np.zeros(3), np.zeros(3)
        self.b1d_dot, self.b1d_2dot = np.zeros(3), np.zeros(3)

        self.trajectory_started  = False
        self.trajectory_complete = False
        
        # Manual mode:
        self.manual_mode = False
        self.manual_mode_init = False
        self.init_b1d = True
        self.x_offset = np.zeros(3)
        self.yaw_offset = 0.0

        # Take-off:
        self.takeoff_end_height = -0.5  # [m]
        self.takeoff_velocity = -0.05  # [m/s]

        # Landing:
        self.landing_velocity = 1.  # [m/s]
        self.landing_motor_cutoff_height = -0.25  # [m]

        # Circle:
        self.num_circles = 2
        self.circle_radius = 0.7
        self.circle_linear_v = 0.4
        self.circle_W = 0.4
        
        # Eight-shaped curve:
        """ Quadrotor control with exponential term:
         ex 1, self.eight_A1 = 0.2, self.eight_A2 = 0.3, self.eight_w = 1.3, self.eight_R_xy = 0.7
         ex 2, self.eight_A1 = 0.7, self.eight_A2 = 0.8, self.eight_w = 0.8, self.eight_R_xy = 0.5
         ex 3, self.eight_A1 = 0.9, self.eight_A2 = 1., self.eight_w = 0.7, self.eight_R_xy = 0.3 """

        self.num_of_eights = 2
        self.eight_A1 = 1.0
        self.eight_A2 = 0.6
        self.eight_T = 10.0 # the period of the cycle [sec]
        self.eight_w1 = 2*np.pi/self.eight_T # w = 2*pi*t/T 
        self.eight_w2 = self.eight_w1
        self.eight_alt_d = -0.3
        self.eight_R_z = 1.#0.1
        self.eight_R_xy = 1.#0.7

    
    def get_desired(self, state, mode):
        # De-normalize state: [-1, 1] -> [max, min]
        self.x_norm, self.v_norm, self.W_norm = state[0:3], state[3:6], state[15:18]
        self.x, self.v, self.R, self.W = state_de_normalization(state, self.x_lim, self.v_lim, self.W_lim)

        # Generate desired traj: 
        if mode == self.mode:
            self.is_mode_changed = False
        else:
            self.is_mode_changed = True
            self.mode = mode
            self.mark_traj_start()
        self.calculate_desired()

        # Normalization goal state vectors: [max, min] -> [-1, 1]
        self.xd_norm = self.xd/self.x_lim
        self.vd_norm = self.vd/self.v_lim
        self.Wd_norm = self.Wd/self.W_lim

        return self.xd_norm, self.vd_norm, self.b1d, self.b3d, self.Wd_norm


    def get_desired_geometric_controller(self):

        return self.xd, self.vd, self.xd_2dot, self.xd_3dot, self.xd_4dot, \
               self.b1d, self.b1d_dot, self.b1d_2dot
    
    
    def calculate_desired(self):
        # Rotation on e3 axis
        def R_e3(theta):
            return np.array([[cos(theta), -sin(theta), 0.],
                             [sin(theta),  cos(theta), 0.],
                             [        0.,          0., 1.]])

        if self.manual_mode:
            self.manual()
            return
        
        if self.mode == 0 or self.mode == 1:  # idle and warm-up
            if self.init_b1d == True:
                self.set_desired_states_to_zero()
                b1d_temp = self.get_current_b1()
                theta_b1d = np.random.uniform(size=1,low=np.pi/6, high=np.pi/4) 
                self.b1d = R_e3(theta_b1d) @ b1d_temp 
                # print(theta_b1d, b1d_temp, self.b1d)
                self.init_b1d = False
        elif self.mode == 2:  # take-off
            self.takeoff()
        elif self.mode == 3:  # land
            self.land()
        elif self.mode == 4:  # stay
            self.stay()
        elif self.mode == 5:  # circle
            self.circle()
        elif self.mode == 6:  #  eight-shaped curve
            self.eight_shaped_curve()


    def mark_traj_start(self):
        self.trajectory_started  = False
        self.trajectory_complete = False

        self.manual_mode_init = False
        self.manual_mode = False
        self.is_landed = False

        self.t0 = datetime.datetime.now()
        self.t = 0.0
        self.t_traj = 0.0

        self.x_offset = np.zeros(3)
        self.yaw_offset = 0.
        # self.yaw = np.random.uniform(size=1,low=-np.pi, high=np.pi) 
        self.init_b1d = True
        self.update_initial_state()
        self.reset_integral_terms()


    def mark_traj_end(self, switch_to_manual=False):
        self.trajectory_complete = True

        if switch_to_manual:
            self.manual_mode = True


    def update_initial_state(self):
        self.x_init = np.copy(self.x)
        self.v_init = np.copy(self.v)
        self.R_init = np.copy(self.R)
        self.W_init = np.copy(self.W)
        self.b1_init = self.get_current_b1()
        self.theta_init = np.arctan2(self.b1_init[1], self.b1_init[0])


    def set_desired_states_to_zero(self):
        self.xd, self.vd, self.Wd = np.zeros(3), np.zeros(3), np.zeros(3)
        self.b1d = np.array([1.,0.,0.]) # desired heading direction
        self.b3d = np.array([0.,0.,1.])

    
    def set_desired_states_to_current(self):
        self.xd = np.copy(self.x)
        self.vd = np.copy(self.v)
        self.b1d = np.array([1.,0.,0.]) #TODO: self.get_current_b1()

    
    def get_current_b1(self):
        b1 = self.R.dot(self.e1)
        theta = np.arctan2(b1[1], b1[0])
        return np.array([np.cos(theta), np.sin(theta), 0.])


    def update_current_time(self):
        if self.is_realtime == True:
            t_now = datetime.datetime.now()
            self.t = (t_now - self.t0).total_seconds()
        else:
            self.t = self.t + self.dt


    def manual(self):
        if not self.manual_mode_init:
            self.set_desired_states_to_current()
            self.update_initial_state()
            self.reset_integral_terms()

            self.manual_mode_init = True
            self.x_offset = np.zeros(3)
            self.yaw_offset = 0.

            # print('Switched to manual mode')
        
        self.xd = self.x_init + self.x_offset
        self.vd = np.zeros(3)

        theta = self.theta_init + self.yaw_offset
        self.b1d = np.array([1.,0.,0.]) #TODO: np.array([np.cos(theta), np.sin(theta), 0.0])


    def takeoff(self):
        if not self.trajectory_started:
            self.set_desired_states_to_zero()
            self.reset_integral_terms()

            # Take-off starts from the current horizontal position:
            self.xd[0] = self.x[0]
            self.xd[1] = self.x[1]
            self.x_init = self.x

            self.t_traj = (self.takeoff_end_height - self.x[2]) / self.takeoff_velocity

            # Set the take-off yaw to the current yaw:
            self.b1d = np.array([1.,0.,0.]) #TODO: self.get_current_b1()

            self.trajectory_started = True

        self.update_current_time()

        if self.t < self.t_traj:
            self.xd[2] = self.x_init[2] + self.takeoff_velocity * self.t 
            self.xd_2dot[2] = self.takeoff_velocity
        else:
            if self.waypoint_reached(self.xd, self.x, 0.04):
                self.xd[2] = self.takeoff_end_height
                self.vd[2] = 0.

                if not self.trajectory_complete:
                    print('Takeoff complete\nSwitching to manual mode')
                
                self.mark_traj_end(True)


    def waypoint_reached(self, waypoint, current, radius):
        delta = waypoint - current
        
        if abs(np.linalg.norm(delta) < radius):
            return True
        else:
            return False
        

    def land(self):
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.reset_integral_terms()
            self.t_traj = (self.landing_motor_cutoff_height - self.x[2]) / self.landing_velocity

            # Set the take-off yaw to the current yaw:
            self.b1d = np.array([1.,0.,0.]) #TODO: self.get_current_b1()

            self.trajectory_started = True

        self.update_current_time()

        if self.t < self.t_traj:
            self.xd[2] = self.x_init[2] + self.landing_velocity * self.t
            self.xd_2dot[2] = self.landing_velocity
        else:
            if self.x[2] > self.landing_motor_cutoff_height:
                self.xd[2] = self.landing_motor_cutoff_height
                self.vd[2] = 0.

                if not self.trajectory_complete:
                    print('Landing complete')

                self.mark_traj_end(False)
                self.is_landed = True
            else:
                self.xd[2] = self.landing_motor_cutoff_height
                self.vd[2] = self.landing_velocity

            
    def stay(self):
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.trajectory_started = True
        
        self.mark_traj_end(True)


    def circle(self):
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.reset_integral_terms()
            self.trajectory_started = True

            self.circle_center = np.copy(self.x)
            self.t_traj = self.circle_radius / self.circle_linear_v \
                        + self.num_circles * 2 * np.pi / self.circle_W

        self.update_current_time()

        if self.t < self.circle_radius / self.circle_linear_v:
            self.xd[0] = self.circle_center[0] + self.circle_linear_v * self.t
            self.vd[0] = self.circle_linear_v

        elif self.t < self.t_traj:
            circle_W = self.circle_W
            circle_radius = self.circle_radius

            t = self.t - circle_radius / self.circle_linear_v
            th = circle_W * t

            circle_W2 = circle_W * circle_W
            circle_W3 = circle_W2 * circle_W
            circle_W4 = circle_W3 * circle_W

            # x-axis:
            self.xd[0] = circle_radius * np.cos(th) + self.circle_center[0]
            self.vd[0] = -circle_radius * circle_W * np.sin(th)
            self.xd_2dot[0] = -circle_radius * circle_W2 * np.cos(th)
            self.xd_3dot[0] =  circle_radius * circle_W3 * np.sin(th)
            self.xd_4dot[0] =  circle_radius * circle_W4 * np.cos(th)

            # y-axis:
            self.xd[1] = circle_radius * np.sin(th) + self.circle_center[1]
            self.vd[1] = circle_radius * circle_W * np.cos(th)
            self.xd_2dot[1] = -circle_radius * circle_W2 * np.sin(th)
            self.xd_3dot[1] = -circle_radius * circle_W3 * np.cos(th)
            self.xd_4dot[1] =  circle_radius * circle_W4 * np.sin(th)

            # yaw-axis:
            w_b1d = 0.01*np.pi
            th_b1d = w_b1d * t
            self.b1d = np.array([np.cos(th_b1d), np.sin(th_b1d), 0])
            self.b1d_dot = np.array([- w_b1d * np.sin(th_b1d), \
                w_b1d * np.cos(th_b1d), 0.0])
            self.b1d_2dot = np.array([- w_b1d * w_b1d * np.cos(th_b1d),
                w_b1d * w_b1d * np.sin(th_b1d), 0.0])
            '''
            self.b1d = np.array([1.,0.,0.]) 
            self.b1d_dot, self.b1d_2dot = np.zeros(3), np.zeros(3)
            '''
        else:
            self.mark_traj_end(True)


    def eight_shaped_curve(self):
        """
        8 shaped curve(Lissajous Curve): https://www.youtube.com/watch?v=7-v6wruWGME 
        xd_1(t) = A1*cos(w1*t), xd_2(t) = A2*sin(w2*t), where A1/A2 are amplitude and w1/w2 are frequency.
        Set possible positive values of A1, A2, w1 and w2.
        ex, xd_1(t) = 3cos(3t + 1), xd_2(t) = sin(5t)
        """
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.reset_integral_terms()
            self.trajectory_started = True

            self.eight_shaped_center = np.copy(self.x)
            self.t_traj = self.num_of_eights * self.eight_T

            self.eight_R_xy_2 = self.eight_R_xy**2
            self.eight_R_xy_3 = self.eight_R_xy**3
            self.eight_R_xy_4 = self.eight_R_xy**4
            
            self.eight_w1_2 = self.eight_w1**2
            self.eight_w1_3 = self.eight_w1**3
            self.eight_w1_4 = self.eight_w1**4

            self.eight_w2_2 = self.eight_w2**2
            self.eight_w2_3 = self.eight_w2**3
            self.eight_w2_4 = self.eight_w2**4

        self.update_current_time()

        if self.t < self.t_traj:
            # x1 commands
            self.xd[0] = self.eight_A2*(-sin(self.t*2.*self.eight_w2) * (np.exp(-self.eight_R_xy * self.t) - 1.)) + self.eight_shaped_center[0]
            self.vd[0] = self.eight_A2*(self.eight_R_xy * np.exp(-self.eight_R_xy * self.t) * sin(self.t*2.*self.eight_w2) \
                                        - 2.*self.eight_w2 * cos(self.t*2.*self.eight_w2) * (np.exp(-self.eight_R_xy * self.t) - 1.))
            self.xd_2dot[0] = self.eight_A2*(2.*self.eight_w2_2 * sin(self.t*2.*self.eight_w2) * (np.exp(-self.eight_R_xy * self.t) - 1.) \
                                             - self.eight_R_xy_2 * np.exp(-self.eight_R_xy * self.t) * sin(self.t*2.*self.eight_w2) \
                                             + 2.*self.eight_R_xy*2.*self.eight_w2 * np.exp(-self.eight_R_xy * self.t) * cos(self.t*2.*self.eight_w2))
            self.xd_3dot[0]  = self.eight_A2*(self.eight_R_xy_3 * np.exp(-self.eight_R_xy * self.t) * sin(self.t*2.*self.eight_w2) \
                                              + 2.*self.eight_w2_3 * cos(self.t*2.*self.eight_w2) * (np.exp(-self.eight_R_xy * self.t) - 1.) \
                                              - 3.*self.eight_R_xy_2*2.*self.eight_w2 * np.exp(-self.eight_R_xy * self.t) * cos(self.t*2.*self.eight_w2) \
                                              - 3.*self.eight_R_xy*2.*self.eight_w2_2 * np.exp(-self.eight_R_xy * self.t) * sin(self.t*2.*self.eight_w2))
            self.xd_4dot[0]  = self.eight_A2*(6.*self.eight_R_xy_2*2.*self.eight_w2_2 * np.exp(-self.eight_R_xy * self.t) * sin(self.t*2.*self.eight_w2) \
                                              - 2.*self.eight_w2_4 * sin(self.t*2.*self.eight_w2) * (np.exp(-self.eight_R_xy * self.t) - 1.) \
                                              - self.eight_R_xy_4 * np.exp(-self.eight_R_xy * self.t) * sin(self.t*2.*self.eight_w2) \
                                              - 4.*self.eight_R_xy*2.*self.eight_w2_3 * np.exp(-self.eight_R_xy * self.t) * cos(self.t*2.*self.eight_w2) \
                                              + 4.*self.eight_R_xy_3*2.*self.eight_w2 * np.exp(-self.eight_R_xy * self.t) * cos(self.t*2.*self.eight_w2))

            # x2 commands
            self.xd[1] = self.eight_A1 * (-(np.exp(-self.eight_R_xy * self.t) - 1.) * (cos(self.t * self.eight_w1) - 1.)) + self.eight_shaped_center[1]
            self.vd[1] = self.eight_A1 * (self.eight_R_xy * np.exp(-self.eight_R_xy * self.t) * (cos(self.t * self.eight_w1) - 1.) \
                                          + self.eight_w1 * sin(self.t * self.eight_w1) * (np.exp(-self.eight_R_xy * self.t) - 1.))
            self.xd_2dot[1]  = self.eight_A1 * (self.eight_w1_2 * cos(self.t * self.eight_w1) * (np.exp(-self.eight_R_xy * self.t) - 1.) \
                                                - self.eight_R_xy_2 * np.exp(-self.eight_R_xy * self.t) * (cos(self.t * self.eight_w1) - 1.) \
                                                - 2.*self.eight_R_xy * self.eight_w1 * np.exp(-self.eight_R_xy * self.t) * sin(self.t * self.eight_w1))
            self.xd_3dot[1]  = self.eight_A1 * (self.eight_R_xy_3 * np.exp(-self.eight_R_xy *self.t) * (cos(self.t * self.eight_w1) - 1.) \
                                                - self.eight_w1_3 * sin(self.t * self.eight_w1) * (np.exp(-self.eight_R_xy * self.t) - 1.) \
                                                - 3.*self.eight_R_xy * self.eight_w1_2 * np.exp(-self.eight_R_xy * self.t) * cos(self.t * self.eight_w1) \
                                                + 3.*self.eight_R_xy_2 * self.eight_w1 * np.exp(-self.eight_R_xy * self.t) * sin(self.t * self.eight_w1))
            self.xd_4dot[1]  = self.eight_A1 * (6.*self.eight_R_xy_2 * self.eight_w1_2 * np.exp(-self.eight_R_xy * self.t) * cos(self.t * self.eight_w1) \
                                                - self.eight_w1_4 * cos(self.t * self.eight_w1) * (np.exp(-self.eight_R_xy * self.t) - 1.) \
                                                - self.eight_R_xy_4 * np.exp(-self.eight_R_xy * self.t) * (cos(self.t * self.eight_w1) - 1.) \
                                                + 4.*self.eight_R_xy * self.eight_w1_3 * np.exp(-self.eight_R_xy * self.t) * sin(self.t*self.eight_w1) \
                                                - 4.*self.eight_R_xy_3 * self.eight_w1 * np.exp(-self.eight_R_xy * self.t) * sin(self.t*self.eight_w1))

            # z Commads
            self.xd[2] = self.eight_shaped_center[2]
            self.vd[2] = 0.
            # self.xd[2] = self.eight_alt_d * (1. - np.exp(-self.eight_R_z*self.t)) + self.eight_shaped_center[2]
            # self.vd[2] = self.eight_alt_d * -self.eight_R_z * -np.exp(-self.eight_R_z*self.t)
            # self.xd_2dot[2] = self.eight_alt_d *  self.eight_R_z**2 * -np.exp(-self.eight_R_z*self.t)
            # self.xd_3dot[2] = self.eight_alt_d * -self.eight_R_z**3 * -np.exp(-self.eight_R_z*self.t)
            # self.xd_4dot[2] = self.eight_alt_d *  self.eight_R_z**4 * -np.exp(-self.eight_R_z*self.t)

            # yaw-axis:
            w_b1d = 0.05*np.pi
            th_b1d = w_b1d * self.t
            self.b1d = np.array([np.cos(th_b1d), np.sin(th_b1d), 0])
            self.b1d_dot = np.array([-w_b1d * np.sin(th_b1d), w_b1d * np.cos(th_b1d), 0.0])
            self.b1d_2dot = np.array([-w_b1d * w_b1d * np.cos(th_b1d), w_b1d * w_b1d * np.sin(th_b1d), 0.0])
            '''
            self.b1d = np.array([1.,0.,0.]) 
            self.b1d_dot, self.b1d_2dot = np.zeros(3), np.zeros(3)
            '''
        else:
            self.mark_traj_end(True)
            

    def reset_integral_terms(self):
        # Reset integral terms:
        self.eIx  = np.zeros(3)
        self.eIb1 = 0.
        self.eIX.set_zero() # Set all integrals to zero
        self.eIR.set_zero()
    

    def get_error_state(self, framework):
        # Normalized error obs:
        ex_norm = self.x_norm - self.xd_norm # position error
        ev_norm = self.v_norm - self.vd_norm # velocity error
        eW_norm = self.W_norm - self.Wd_norm # ang vel error

        # Compute yaw angle error: 
        b1 = self.R @ np.array([1.,0.,0.])
        b2 = self.R @ np.array([0.,1.,0.])
        b3 = self.R @ np.array([0.,0.,1.])
        b1c = -(hat(b3) @ hat(b3)) @ self.b1d # desired b1
        eb1 = norm_ang_btw_two_vectors(b1c, b1) # b1 error, [-1, 1) # -np.dot(b1c,b2)/np.pi
        # wc = hat(b1c)@b1c_dot
        # wc3 = b3@wc
        # ewy = eW_norm[2] - self.wc3
        
        # Update integral terms: 
        self.eIX.integrate(-self.alpha*self.eIX.error + ex_norm*self.x_lim, self.dt) 
        self.eIx = clip(self.eIX.error/self.eIx_lim, -self.sat_sigma, self.sat_sigma)
        self.eIR.integrate(-self.beta*self.eIR.error + eb1*np.pi, self.dt) # b1 integral error
        self.eIb1 = clip(self.eIR.error/self.eIb1_lim, -self.sat_sigma, self.sat_sigma)

        if framework in ("CTDE","DTDE"):
            # Agent1's obs:
            ew12 = eW_norm[0]*b1 + eW_norm[1]*b2
            obs_1 = np.concatenate((ex_norm, ev_norm, b3, ew12, self.eIx), axis=None)
            # Agent2's obs:
            eW3_norm = eW_norm[2]
            obs_2 = np.concatenate((b1, eW3_norm, eb1, self.eIb1), axis=None)
            error_obs_n = [obs_1, obs_2]
        elif framework == "SARL":
            # Single-agent's obs:
            R_vec = self.R.reshape(9, 1, order='F').flatten()
            obs = np.concatenate((ex_norm, ev_norm, R_vec, eW_norm, self.eIx, eb1, self.eIb1), axis=None)
            error_obs_n = [obs]
        error_state = (ex_norm, ev_norm, eW_norm, self.eIx, eb1, self.eIb1)
        
        return error_obs_n, error_state