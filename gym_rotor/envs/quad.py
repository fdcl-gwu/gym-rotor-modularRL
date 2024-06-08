import copy
import numpy as np
from numpy import interp
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.random import uniform 
import random
from math import cos, sin, atan2, sqrt, pi
from scipy.integrate import odeint, solve_ivp
from scipy.spatial.transform import Rotation

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gym_rotor.envs.quad_utils import *
from typing import Optional
import args_parse

class QuadEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None): 
        # Hyperparameters:
        parser = args_parse.create_parser()
        args = parser.parse_args()

        # Quadrotor parameters:
        self.m = 2.15 # mass of quad, [kg]
        self.d = 0.23 # arm length, [m]
        self.J = np.diag([0.022, 0.022, 0.035]) # inertia matrix of quad, [kg m2]
        self.c_tf = 0.0135 # torque-to-thrust coefficients
        self.c_tw = 1.8 # thrust-to-weight coefficients
        self.g = 9.81 # standard gravity

        # Force and Moment:
        self.f = self.m * self.g # magnitude of total thrust to overcome  
                                 # gravity and mass (No air resistance), [N]
        self.hover_force = self.m * self.g / 4.0 # thrust magnitude of each motor, [N]
        self.min_force = 0.5 # minimum thrust of each motor, [N]
        self.max_force = self.c_tw * self.hover_force # maximum thrust of each motor, [N]
        self.avrg_act = (self.min_force+self.max_force)/2.0 
        self.scale_act = self.max_force-self.avrg_act # actor scaling

        self.f1 = self.hover_force # thrust of each 1st motor, [N]
        self.f2 = self.hover_force # thrust of each 2nd motor, [N]
        self.f3 = self.hover_force # thrust of each 3rd motor, [N]
        self.f4 = self.hover_force # thrust of each 4th motor, [N]
        self.M  = np.zeros(3) # magnitude of moment on quadrotor, [Nm]

        self.fM = np.zeros((4, 1)) # Force-moment vector
        self.forces_to_fM = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [0.0, -self.d, 0.0, self.d],
            [self.d, 0.0, -self.d, 0.0],
            [-self.c_tf, self.c_tf, -self.c_tf, self.c_tf]
        ]) # Conversion matrix of forces to force-moment 
        self.fM_to_forces = np.linalg.inv(self.forces_to_fM)

        # Simulation parameters:
        self.freq = 200 # frequency [Hz]
        self.dt = 1./self.freq # discrete timestep, t(2) - t(1), [sec]
        self.ode_integrator = "solve_ivp" # or "euler", ODE solvers
        self.R2D = 180./pi # [rad] to [deg]
        self.D2R = pi/180. # [deg] to [rad]
        self.e1 = np.array([1.,0.,0.])
        self.e2 = np.array([0.,1.,0.])
        self.e3 = np.array([0.,0.,1.])
        self.use_UDM = args.use_UDM # uniform domain randomization for sim-to-real transfer
        self.UDM_percentage = args.UDM_percentage

        # Coefficients in reward function:
        self.framework = args.framework
        self.reward_alive = 0.  # ≥ 0 is a bonus value earned by the agent for staying alive
        self.reward_crash = -1. # Out of boundary or crashed!
        self.Cx = args.Cx
        self.CIx = args.CIx
        self.Cv = args.Cv
        self.Cb1 = args.Cb1
        self.CIb1 = args.CIb1
        self.CW = args.Cw12
        self.reward_min = -np.ceil(self.Cx+self.CIx+self.Cv+self.Cb1+self.CIb1+self.CW)
        if self.framework in ("CTDE","DTDE"):
            # Agent1's reward:
            self.Cw12 = args.Cw12
            self.reward_min_1 = -np.ceil(self.Cx+self.CIx+self.Cv+self.Cw12)
            # Agent2's reward:
            self.CW3 = args.CW3
            self.reward_min_2 = -np.ceil(self.Cb1+self.CW3+self.CIb1)
        
        # Integral terms:
        self.sat_sigma = 1.
        self.eIX = IntegralErrorVec3() # Position integral error
        self.eIR = IntegralError() # Attitude integral error
        self.eIX.set_zero() # Set all integrals to zero
        self.eIR.set_zero()

        # Commands:
        self.xd  = np.array([0.,0.,0.]) # desired tracking position command, [m] 
        self.vd  = np.array([0.,0.,0.]) # [m/s]
        self.b1d = np.array([1.,0.,0.]) # desired heading direction        
        self.Wd  = np.eye(3) # desired angular velocity [rad/s]

        # Limits of states:
        self.x_lim = 1.0 # [m]
        self.v_lim = 4.0 # [m/s]
        self.W_lim = 2*pi # [rad/s]
        self.euler_lim = 85 # [deg]
        self.low = np.concatenate([-self.x_lim * np.ones(3),  
                                   -self.v_lim * np.ones(3),
                                   -np.ones(9),
                                   -self.W_lim * np.ones(3)])
        self.high = np.concatenate([self.x_lim * np.ones(3),  
                                    self.v_lim * np.ones(3),
                                    np.ones(9),
                                    self.W_lim * np.ones(3)])

        # Observation space:
        self.observation_space = spaces.Box(
            low=self.low, 
            high=self.high, 
            dtype=np.float64
        )

        # Action space:
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(4,),
            dtype=np.float64
        ) 

        # Init:
        self.state = None
        self.viewer = None
        self.render_index = 1 


    def step(self, action):
        # Action:
        action = self.action_wrapper(action)

        # State vec: (x[0:3]; v[3:6]; R_vec[6:15]; W[15:18])
        state = copy.deepcopy(self.state)
                 
        # Observation:
        obs = self.observation_wrapper(state)

        # Reward function:
        reward = self.reward_wrapper(obs)
        if self.framework in ("CTDE","DTDE"):
            reward[0] = interp(reward[0], [self.reward_min_1, 0.], [0., 1.]) # linear interpolation [0,1]
            reward[1] = interp(reward[1], [self.reward_min_2, 0.], [0., 1.]) # linear interpolation [0,1]
        elif self.framework == "SARL":
            reward[0] = interp(reward[0], [self.reward_min, 0.], [0., 1.]) # linear interpolation [0,1]  

        # Terminal condition:
        done = self.done_wrapper(obs)
        if done[0]: # Out of boundary or crashed!
            reward[0] = self.reward_crash
        if self.framework in ("CTDE","DTDE"):
            if done[1]: # Out of boundary or crashed!
                reward[1] = self.reward_crash

        return obs, reward, done, False, {}


    def reset(self, env_type='train',
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        # Domain randomization:
        self.set_random_parameters(env_type) if self.use_UDM else None

        # Reset states
        self.state = np.array(np.zeros(18))

        # Initial state error:
        self.sample_init_error(env_type)

        # x, position:
        self.state[0:3] = uniform(size=3,low=-self.init_x,high=self.init_x) 

        # v, velocity:
        self.state[3:6] = uniform(size=3,low=-self.init_v,high=self.init_v) 

        # W, angular velocity:
        self.state[15:18] = uniform(size=3,low=-self.init_W,high=self.init_W) 

        # R, attitude:
        roll_pitch = uniform(size=2,low=-self.init_R,high=self.init_R)
        euler = np.concatenate((roll_pitch, self.yaw), axis=None)
        R = Rotation.from_euler('xyz', euler, degrees=False).as_matrix()
        # Compute current b1:
        b1 = R.dot(self.e1)
        theta = np.arctan2(b1[1], b1[0])
        self.b1d = np.array([np.cos(theta), np.sin(theta), 0.]) 
        # Re-orthonormalize:
        if not isRotationMatrix(R):
            U, s, VT = psvd(R)
            R = U @ VT.T
        self.state[6:15] = R.reshape(9, 1, order='F').flatten()
        
        # Reset forces & moments:
        self.f  = self.m * self.g
        self.f1 = self.hover_force
        self.f2 = self.hover_force
        self.f3 = self.hover_force
        self.f4 = self.hover_force
        self.M  = np.zeros(3)

        # Integral terms:
        self.eIX.set_zero() # Set all integrals to zero
        self.eIR.set_zero()

        # for drawing real-time plots:
        self.t = 0
        self.cmd_count = 0

        return self.state


    def action_wrapper(self, action):
        # Linear scale, [-1, 1] -> [min_act, max_act] 
        action = (
            self.scale_act * action + self.avrg_act
            ).clip(self.min_force, self.max_force)

        # Saturated thrust of each motor:
        self.f1 = action[0]
        self.f2 = action[1]
        self.f3 = action[2]
        self.f4 = action[3]

        # Convert each forces to force-moment:
        self.fM = self.forces_to_fM @ action
        self.f = self.fM[0]   # [N]
        self.M = self.fM[1:4] # [Nm]  

        return action


    def observation_wrapper(self, state):
        # Decomposing state vectors:
        x, v, R, W = state_decomposition(state)
        R_vec = R.reshape(9, 1, order='F').flatten()
        current_state = np.concatenate((x, v, R_vec, W), axis=0)

        # Solve ODEs:
        if self.ode_integrator == "euler": # solve w/ Euler's Method
            # Equations of motion of the quadrotor UAV
            x_dot = v
            v_dot = self.g*self.e3 - self.f*R@self.e3/self.m
            R_vec_dot = (R@hat(W)).reshape(9, 1, order='F')
            W_dot = inv(self.J)@(-hat(W)@self.J@W + self.M)
            state_dot = np.concatenate([x_dot.flatten(), 
                                        v_dot.flatten(),                                                                          
                                        R_vec_dot.flatten(),
                                        W_dot.flatten()])
            next_state = current_state + state_dot * self.dt
        elif self.ode_integrator == "solve_ivp": # solve w/ 'solve_ivp' Solver
            # method = 'RK45', 'DOP853', 'BDF', 'LSODA', ...
            sol = solve_ivp(self.EoM, [0, self.dt], current_state, method='DOP853')
            next_state = sol.y[:,-1]

        # TODO: Add sensor noise

        # Next state vec: (x_next[0:3]; v_next[3:6]; R_next[6:15]; W_next[15:18])
        self.state = next_state

        return self.state
    

    def reward_wrapper(self, obs):
        # Decomposing state vectors
        x, v, R, W = state_decomposition(obs)

        # Reward function coefficients:
        Cx = self.Cx # pos coef.
        Cv = self.Cv # vel coef.
        Cb1 = self.Cb1 # heading coef.
        CW = self.CW # ang_vel coef.

        # Errors:
        eX = x - self.xd # position error
        eV = v - self.vd # velocity error
        eb1 = norm_ang_btw_two_vectors(self.b1d, get_current_b1(R)) # heading errors

        # Reward function:
        reward_eX = -Cx*(norm(eX, 2)**2) 
        reward_eV = -Cv*(norm(eV, 2)**2)
        reward_eb1 = -Cb1*(abs(eb1))
        reward_eW = -CW*(norm(W, 2)**2)

        reward = self.reward_alive + (reward_eX + reward_eb1 + reward_eV + reward_eW)
        #reward *= 0.1 # rescaled by a factor of 0.1

        return reward


    def done_wrapper(self, obs):
        # Decomposing state vectors
        x, v, R, W = state_decomposition(obs)

        # Convert rotation matrix to Euler angles:
        euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
        #eulerAngles = rotationMatrixToEulerAngles(R) * self.R2D

        done = False
        done = bool(
               (abs(x) >= self.x_lim).any() # [m]
            or (abs(v) >= self.v_lim).any() # [m/s]
            or (abs(W) >= self.W_lim).any() # [rad/s]
            or abs(euler[0]) >= self.euler_lim # phi
            or abs(euler[1]) >= self.euler_lim # theta
        )

        return done


    def EoM(self, t, state):
        # Decomposing state vectors
        x, v, R, W = state_decomposition(state)

        # Equations of motion of the quadrotor UAV
        x_dot = v
        v_dot = self.g*self.e3 - self.f*R@self.e3/self.m
        R_vec_dot = (R@hat(W)).reshape(1, 9, order='F')
        W_dot = inv(self.J)@(-hat(W)@self.J@W + self.M)
        state_dot = np.concatenate([x_dot.flatten(), 
                                    v_dot.flatten(),                                                                          
                                    R_vec_dot.flatten(),
                                    W_dot.flatten()])

        return np.array(state_dot)


    def sample_init_error(self, env_type='train'):
        if env_type == 'train':
            # Spawning at the origin position and at zero angle (w/ random linear and angular velocity).
            if random.random() < 0.2: # 20% of the training
                self.init_x = 0.0 # initial pos error,[m]
                self.init_R = 0 * self.D2R  # ±0 deg 
                self.yaw = 0.
                self.init_v = 0. # initial vel error, [m/s]
                self.init_W = 0. # initial ang vel error, [rad/s]
            else:
                self.init_x = 0.5 # initial pos error,[m]
                self.init_R = 50 * self.D2R  # ±50 deg 
                self.yaw = uniform(size=1,low=-pi, high=pi) 
                self.init_v = self.v_lim*0.5 # 50%; initial vel error, [m/s]
                self.init_W = self.W_lim*0.5 # 50%; initial ang vel error, [rad/s]
        elif env_type == 'eval':
            self.init_x = 0.2 # initial pos error,[m]
            self.init_v = self.v_lim*0.1 # initial vel error, [m/s]
            self.init_R = 10 * self.D2R # ±10 deg
            self.init_W = self.W_lim*0.1 # initial ang vel error, [rad/s]
            self.yaw = uniform(size=1,low=-pi/5, high=pi/5) 


    def set_random_parameters(self, env_type='train'):
        # Nominal quadrotor parameters:
        self.m = 2.15 # mass of quad, [kg]
        self.d = 0.23 # arm length, [m]
        J1, J2, J3 = 0.022, 0.022, 0.035
        self.J = np.diag([J1, J2, J3]) # inertia matrix of quad, [kg m2]
        self.c_tf = 0.0135 # torque-to-thrust coefficients
        self.c_tw = 2.2 # thrust-to-weight coefficients

        if env_type == 'train':
            uncertainty_range = self.UDM_percentage/100
            # Quadrotor parameters:
            m_range = self.m * uncertainty_range
            d_range = self.d * uncertainty_range
            J1_range = J1 * uncertainty_range
            J3_range = J3 * uncertainty_range
            # c_tf_range = self.c_tf * uncertainty_range
            # c_tw_range = self.c_tw * uncertainty_range

            self.m = uniform(low=(self.m - m_range), high=(self.m + m_range)) # [kg]
            self.d = uniform(low=(self.d - d_range), high=(self.d + d_range)) # [m]
            J1 = uniform(low=(J1 - J1_range), high=(J1 + J1_range))
            J2 = J1 
            J3 = uniform(low=(J3 - J3_range), high=(J3 + J3_range))
            self.J  = np.diag([J1, J2, J3]) # [kg m2]
            # self.c_tf = uniform(low=(self.c_tf - c_tf_range), high=(self.c_tf + c_tf_range))
            # self.c_tw = uniform(low=(self.c_tw - c_tw_range), high=(self.c_tw + c_tw_range))
                        
        # Force and Moment:
        self.f = self.m * self.g # magnitude of total thrust to overcome  
                                    # gravity and mass (No air resistance), [N]
        self.hover_force = self.m * self.g / 4.0 # thrust magnitude of each motor, [N]
        self.min_force = 0.5 # minimum thrust of each motor, [N]
        self.max_force = self.c_tw * self.hover_force # maximum thrust of each motor, [N]
        self.fM = np.zeros((4, 1)) # Force-moment vector
        self.forces_to_fM = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [0.0, -self.d, 0.0, self.d],
            [self.d, 0.0, -self.d, 0.0],
            [-self.c_tf, self.c_tf, -self.c_tf, self.c_tf]
        ]) # Conversion matrix of forces to force-moment 
        self.fM_to_forces = np.linalg.inv(self.forces_to_fM)
        self.avrg_act = (self.min_force+self.max_force)/2.0 
        self.scale_act = self.max_force-self.avrg_act # actor scaling

        # print('m:',f'{self.m:.3f}','d:',f'{self.d:.3f}','J:',f'{J1:.4f}',f'{J3:.4f}','c_tf:',f'{self.c_tf:.4f}','c_tw:',f'{self.c_tw:.3f}')
        

    def get_current_state(self):
        return self.state


    def set_goal_state(self, xd, vd, b1d, Wd):
        self.xd  = xd # desired tracking position command, [m] 
        self.vd  = vd # desired velocity command, [m/s]
        self.b1d = b1d # desired heading direction        
        self.Wd  = Wd # desired angular velocity [rad/s]


    def get_norm_error_state(self, framework):
        # Normalize state vectors: [max, min] -> [-1, 1]
        x_norm, v_norm, R_vec, W_norm = state_normalization(self.state, self.x_lim, self.v_lim, self.W_lim)
        R = R_vec.reshape(3, 3, order='F')

        # Normalize goal state vectors: [max, min] -> [-1, 1]
        xd_norm = self.xd/self.x_lim
        vd_norm = self.vd/self.v_lim
        Wd_norm = self.Wd/self.W_lim

        # Normalized error obs:
        ex_norm = x_norm - xd_norm # norm pos error
        ev_norm = v_norm - vd_norm # norm vel error
        eW_norm = W_norm - Wd_norm # norm ang vel error

        # Compute yaw angle error: 
        b1 = R @ np.array([1.,0.,0.])
        b2 = R @ np.array([0.,1.,0.])
        b3 = R @ np.array([0.,0.,1.])
        b1c = -(hat(b3) @ hat(b3)) @ self.b1d # desired b1
        eb1 = norm_ang_btw_two_vectors(b1c, b1) # b1 error, [-1, 1) # -np.dot(b1c,b2)/np.pi
        
        # Update integral terms: 
        self.eIX.integrate(-self.alpha*self.eIX.error + ex_norm*self.x_lim, self.dt) 
        self.eIx = clip(self.eIX.error/self.eIx_lim, -self.sat_sigma, self.sat_sigma)
        self.eIR.integrate(-self.beta*self.eIR.error + eb1*np.pi, self.dt) # b1 integral error
        self.eIb1 = clip(self.eIR.error/self.eIb1_lim, -self.sat_sigma, self.sat_sigma)

        if framework in ("CTDE","DTDE"):
            # Agent1's obs:
            ew12 = eW_norm[0]*b1 + eW_norm[1]*b2
            obs_1 = np.concatenate((ex_norm, self.eIx, ev_norm, b3, ew12), axis=None)
            # Agent2's obs:
            eW3_norm = eW_norm[2]
            obs_2 = np.concatenate((b1, eb1, self.eIb1, eW3_norm), axis=None)
            error_obs_n = [obs_1, obs_2]
        elif framework == "SARL":
            # Single-agent's obs:
            R_vec = R.reshape(9, 1, order='F').flatten()
            obs = np.concatenate((ex_norm, self.eIx, ev_norm, R_vec, eb1, self.eIb1, eW_norm), axis=None)
            error_obs_n = [obs]
        
        return error_obs_n



    def render(self, mode='human', close=False):
        # https://engcourses-uofa.ca/wp-content/uploads/Visual-Python-VPython-ver-2.pdf
        from vpython import canvas, vector, box, sphere, color, rate, cylinder, arrow, vec, graph, gcurve, gdots

        # Rendering state:
        state_vis = np.copy(self.state)

        # De-normalization state vectors
        x, v, R, W = state_decomposition(state_vis)

        # Quadrotor and goal positions:
        quad_pos = x # [m]
        cmd_pos  = self.xd # [m]
        
        # Heading commands:
        b1d_vis = self.b1d

        # Axis:
        b1, b2, b3 = R@self.e1, R@self.e2, R@self.e3

        # Init:
        if self.viewer is None:
            # Canvas.
            self.viewer = canvas(title='Quadrotor with RL', width=1080, height=480, \
                                 center=vector(0, 0, cmd_pos[2]), background=color.white, \
                                 forward=vector(0.5, 0.3, 0.7), up=vector(0, 0, -1), range=2.0) # forward = view point
            
            # Quad body.
            self.render_quad1 = box(canvas=self.viewer, pos=vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                    axis=vector(b1[0], b1[1], b1[2]), \
                                    length=0.2, height=0.05, width=0.05) # vector(quad_pos[0], quad_pos[1], 0)
            self.render_quad2 = box(canvas=self.viewer, pos=vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                    axis=vector(b2[0], b2[1], b2[2]), \
                                    length=0.2, height=0.05, width=0.05)
            # Rotors.
            rotors_offest = 0.02
            self.render_rotor1 = cylinder(canvas=self.viewer, pos=vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                          axis=vector(rotors_offest*b3[0], rotors_offest*b3[1], rotors_offest*b3[2]), \
                                          radius=0.2, color=color.blue, opacity=0.5)
            self.render_rotor2 = cylinder(canvas=self.viewer, pos=vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                          axis=vector(rotors_offest*b3[0], rotors_offest*b3[1], rotors_offest*b3[2]), \
                                          radius=0.2, color=color.cyan, opacity=0.5)
            self.render_rotor3 = cylinder(canvas=self.viewer, pos=vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                          axis=vector(rotors_offest*b3[0], rotors_offest*b3[1], rotors_offest*b3[2]), \
                                          radius=0.2, color=color.blue, opacity=0.5)
            self.render_rotor4 = cylinder(canvas=self.viewer, pos=vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                          axis=vector(rotors_offest*b3[0], rotors_offest*b3[1], rotors_offest*b3[2]), \
                                          radius=0.2, color=color.cyan, opacity=0.5)

            # Force arrows.
            self.render_force_rotor1 = arrow(pos=vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                             axis=vector(b3[0], b3[1], b3[2]), \
                                             shaftwidth=0.05, color=color.blue)
            self.render_force_rotor2 = arrow(pos=vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                             axis=vector(b3[0], b3[1], b3[2]), \
                                             shaftwidth=0.05, color=color.cyan)
            self.render_force_rotor3 = arrow(pos=vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                             axis=vector(b3[0], b3[1], b3[2]), \
                                             shaftwidth=0.05, color=color.blue)
            self.render_force_rotor4 = arrow(pos=vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                             axis=vector(b3[0], b3[1], b3[2]), \
                                             shaftwidth=0.05, color=color.cyan)
                                    
            # Commands.
            self.render_xd = sphere(canvas=self.viewer, pos=vector(cmd_pos[0], cmd_pos[1], cmd_pos[2]), \
                                     radius=0.07, color=color.red, \
                                     make_trail=True, trail_type='points', interval=10)		
            self.render_b1d = arrow(canvas=self.viewer, pos=vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                     axis=vector(b1d_vis[0], b1d_vis[1], b1d_vis[2]), \
                                     shaftwidth=0.03, color=color.orange)							
            
            # Inertial axis.				
            self.e1_axis = arrow(pos=vector(2.5, -2.5, 0), axis=0.5*vector(1, 0, 0), \
                                 shaftwidth=0.04, color=color.blue)
            self.e2_axis = arrow(pos=vector(2.5, -2.5, 0), axis=0.5*vector(0, 1, 0), \
                                 shaftwidth=0.04, color=color.green)
            self.e3_axis = arrow(pos=vector(2.5, -2.5, 0), axis=0.5*vector(0, 0, 1), \
                                 shaftwidth=0.04, color=color.red)

            # Body axis.				
            self.render_b1_axis = arrow(canvas=self.viewer, pos=vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                        axis=vector(b1[0], b1[1], b1[2]), \
                                        shaftwidth=0.02, color=color.blue, \
                                        make_trail=True, retain=200, interval=10, \
                                        trail_type='points', trail_radius=0.02, trail_color=color.yellow)
            self.render_b2_axis = arrow(canvas=self.viewer, pos=vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                        axis=vector(b2[0], b2[1], b2[2]), \
                                        shaftwidth=0.02, color=color.green)
            self.render_b3_axis = arrow(canvas=self.viewer, pos=vector(quad_pos[0], quad_pos[1], quad_pos[2]), \
                                        axis=vector(b3[0], b3[1], b3[2]), \
                                        shaftwidth=0.02, color=color.red)

            # Floor.
            self.render_floor = box(pos=vector(0,0,0),size=vector(5,5,0.05), axis=vector(1,0,0), \
                                    opacity=0.2, color=color.black)
        
            # Real-time graphing.
            self.gx1 = graph(width=270, height=150, xtitle='t (sec)', ytitle='x1 (m)', \
                       foreground=vec(.5,.5,.5), background=color.white, align='left', fast=False)
                       # xmin=0, xmax=-20, ymin=0, ymax=-20)
            self.gc_x1d = gdots(graph=self.gx1, color=color.red, radius=2.)
            self.gc_x1 = gcurve(graph=self.gx1, color=color.blue, dot=True, dot_radius=2)
            self.gx2 = graph(width=270, height=150, xtitle='t (sec)', ytitle='x2 (m)', \
                       foreground=vec(.5,.5,.5), background=color.white, align='left', fast=False)
            self.gc_x2d = gdots(graph=self.gx2, color=color.red, radius=2.)
            self.gc_x2 = gcurve(graph=self.gx2, color=color.blue, dot=True, dot_radius=2)
            self.gx3 = graph(width=270, height=150, xtitle='t (sec)', ytitle='x3 (m)', \
                       foreground=vec(.5,.5,.5), background=color.white, align='left', fast=False)
            self.gc_x3d = gdots(graph=self.gx3, color=color.red, radius=2.)
            self.gc_x3 = gcurve(graph=self.gx3, color=color.blue, dot=True, dot_radius=2)
            self.gR11 = graph(width=270, height=150, xtitle='t (sec)', ytitle='R11', \
                        foreground=vec(.5,.5,.5), background=color.white, align='left', fast=False)
            self.gc_R11d = gdots(graph=self.gR11, color=color.red, radius=2.)
            self.gc_R11 = gcurve(graph=self.gR11, color=color.blue, dot=True, dot_radius=2)

        # Update visualization component:
        if self.state is None: 
            return None
        
        # Update graphs.
        if self.t == 0:  # delete all graphs
            self.gc_x1d.delete()
            self.gc_x1.delete()
            self.gc_x2d.delete()
            self.gc_x2.delete()
            self.gc_x3d.delete()
            self.gc_x3.delete()
            self.gc_R11d.delete()
            self.gc_R11.delete()
        self.t += self.dt  # update time
        self.cmd_count += 1
        self.gc_x1.plot(self.t, quad_pos[0])
        self.gc_x2.plot(self.t, quad_pos[1])
        self.gc_x3.plot(self.t, quad_pos[2])
        self.gc_R11.plot(self.t, state_vis[6])        
        if self.cmd_count == 1 or self.cmd_count%80 == 0:
            self.gc_x1d.plot(self.t, cmd_pos[0])
            self.gc_x2d.plot(self.t, cmd_pos[1])
            self.gc_x3d.plot(self.t, cmd_pos[2])
            self.gc_R11d.plot(self.t, b1d_vis[0])
        
        # Update quad body.
        self.render_quad1.pos.x = quad_pos[0]
        self.render_quad1.pos.y = quad_pos[1]
        self.render_quad1.pos.z = quad_pos[2]
        self.render_quad2.pos.x = quad_pos[0]
        self.render_quad2.pos.y = quad_pos[1]
        self.render_quad2.pos.z = quad_pos[2]

        self.render_quad1.axis.x = b1[0]
        self.render_quad1.axis.y = b1[1]	
        self.render_quad1.axis.z = b1[2]
        self.render_quad2.axis.x = b2[0]
        self.render_quad2.axis.y = b2[1]
        self.render_quad2.axis.z = b2[2]

        self.render_quad1.up.x = b3[0]
        self.render_quad1.up.y = b3[1]
        self.render_quad1.up.z = b3[2]
        self.render_quad2.up.x = b3[0]
        self.render_quad2.up.y = b3[1]
        self.render_quad2.up.z = b3[2]

        # Update rotors.
        rotors_offest = -0.02
        rotor_pos = 0.5*b1
        self.render_rotor1.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_rotor1.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_rotor1.pos.z = quad_pos[2] + rotor_pos[2]
        rotor_pos = 0.5*b2
        self.render_rotor2.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_rotor2.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_rotor2.pos.z = quad_pos[2] + rotor_pos[2]
        rotor_pos = (-0.5)*b1
        self.render_rotor3.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_rotor3.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_rotor3.pos.z = quad_pos[2] + rotor_pos[2]
        rotor_pos = (-0.5)*b2
        self.render_rotor4.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_rotor4.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_rotor4.pos.z = quad_pos[2] + rotor_pos[2]

        self.render_rotor1.axis.x = rotors_offest*b3[0]
        self.render_rotor1.axis.y = rotors_offest*b3[1]
        self.render_rotor1.axis.z = rotors_offest*b3[2]
        self.render_rotor2.axis.x = rotors_offest*b3[0]
        self.render_rotor2.axis.y = rotors_offest*b3[1]
        self.render_rotor2.axis.z = rotors_offest*b3[2]
        self.render_rotor3.axis.x = rotors_offest*b3[0]
        self.render_rotor3.axis.y = rotors_offest*b3[1]
        self.render_rotor3.axis.z = rotors_offest*b3[2]
        self.render_rotor4.axis.x = rotors_offest*b3[0]
        self.render_rotor4.axis.y = rotors_offest*b3[1]
        self.render_rotor4.axis.z = rotors_offest*b3[2]

        self.render_rotor1.up.x = b2[0]
        self.render_rotor1.up.y = b2[1]
        self.render_rotor1.up.z = b2[2]
        self.render_rotor2.up.x = b2[0]
        self.render_rotor2.up.y = b2[1]
        self.render_rotor2.up.z = b2[2]
        self.render_rotor3.up.x = b2[0]
        self.render_rotor3.up.y = b2[1]
        self.render_rotor3.up.z = b2[2]
        self.render_rotor4.up.x = b2[0]
        self.render_rotor4.up.y = b2[1]
        self.render_rotor4.up.z = b2[2]

        # Update force arrows.
        rotor_pos = 0.5*b1
        self.render_force_rotor1.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_force_rotor1.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_force_rotor1.pos.z = quad_pos[2] + rotor_pos[2]
        rotor_pos = 0.5*b2
        self.render_force_rotor2.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_force_rotor2.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_force_rotor2.pos.z = quad_pos[2] + rotor_pos[2]
        rotor_pos = (-0.5)*b1
        self.render_force_rotor3.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_force_rotor3.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_force_rotor3.pos.z = quad_pos[2] + rotor_pos[2]
        rotor_pos = (-0.5)*b2
        self.render_force_rotor4.pos.x = quad_pos[0] + rotor_pos[0]
        self.render_force_rotor4.pos.y = quad_pos[1] + rotor_pos[1]
        self.render_force_rotor4.pos.z = quad_pos[2] + rotor_pos[2]

        force_offest = -0.05
        self.render_force_rotor1.axis.x = force_offest * self.f1 * b3[0] 
        self.render_force_rotor1.axis.y = force_offest * self.f1 * b3[1]
        self.render_force_rotor1.axis.z = force_offest * self.f1 * b3[2]
        self.render_force_rotor2.axis.x = force_offest * self.f2 * b3[0]
        self.render_force_rotor2.axis.y = force_offest * self.f2 * b3[1]
        self.render_force_rotor2.axis.z = force_offest * self.f2 * b3[2]
        self.render_force_rotor3.axis.x = force_offest * self.f3 * b3[0]
        self.render_force_rotor3.axis.y = force_offest * self.f3 * b3[1]
        self.render_force_rotor3.axis.z = force_offest * self.f3 * b3[2]
        self.render_force_rotor4.axis.x = force_offest * self.f4 * b3[0]
        self.render_force_rotor4.axis.y = force_offest * self.f4 * b3[1]
        self.render_force_rotor4.axis.z = force_offest * self.f4 * b3[2]

        # Update commands.
        self.render_xd.pos.x = cmd_pos[0]
        self.render_xd.pos.y = cmd_pos[1]
        self.render_xd.pos.z = cmd_pos[2]

        axis_offest = 0.9
        self.render_b1d.pos.x = quad_pos[0]
        self.render_b1d.pos.y = quad_pos[1]
        self.render_b1d.pos.z = quad_pos[2]
        self.render_b1d.axis.x = axis_offest * b1d_vis[0] 
        self.render_b1d.axis.y = axis_offest * b1d_vis[1] 
        self.render_b1d.axis.z = axis_offest * b1d_vis[2] 
        
        # Update body axis.
        axis_offest = 0.8
        self.render_b1_axis.pos.x = quad_pos[0]
        self.render_b1_axis.pos.y = quad_pos[1]
        self.render_b1_axis.pos.z = quad_pos[2]
        self.render_b2_axis.pos.x = quad_pos[0]
        self.render_b2_axis.pos.y = quad_pos[1]
        self.render_b2_axis.pos.z = quad_pos[2]
        self.render_b3_axis.pos.x = quad_pos[0]
        self.render_b3_axis.pos.y = quad_pos[1]
        self.render_b3_axis.pos.z = quad_pos[2]

        self.render_b1_axis.axis.x = axis_offest * b1[0] 
        self.render_b1_axis.axis.y = axis_offest * b1[1] 
        self.render_b1_axis.axis.z = axis_offest * b1[2] 
        self.render_b2_axis.axis.x = axis_offest * b2[0] 
        self.render_b2_axis.axis.y = axis_offest * b2[1] 
        self.render_b2_axis.axis.z = axis_offest * b2[2] 
        self.render_b3_axis.axis.x = (axis_offest/2) * b3[0] 
        self.render_b3_axis.axis.y = (axis_offest/2) * b3[1]
        self.render_b3_axis.axis.z = (axis_offest/2) * b3[2]

        # Screen capture:
        """
        if (self.render_index % 5) == 0:
            self.viewer.capture('capture'+str(self.render_index))
        self.render_index += 1        
        """

        rate(60) # FPS

        return True


    def close(self):
        if self.viewer:
            self.viewer = None