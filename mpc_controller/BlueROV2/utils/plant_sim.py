import numpy as np
import casadi as cas
from core.kinematics import get_J1_np, get_J2_np, get_J1, get_J2

class Vehicle_Sim_Utils:
    def __init__(self, params):
        self.p = params

    def get_C_np(self, nu):
        """ Numpy Coriolis Matrix """
        C_rb = np.zeros((6, 6))
        m, Ix, Iy, Iz = self.p.m, self.p.Ix, self.p.Iy, self.p.Iz
        
        C_rb[0, 4] =  m * nu[2];  C_rb[0, 5] = -m * nu[1]
        C_rb[1, 3] = -m * nu[2];  C_rb[1, 5] =  m * nu[0]
        C_rb[2, 3] =  m * nu[1];  C_rb[2, 4] = -m * nu[0]
        C_rb[3, 1] =  m * nu[2];  C_rb[3, 2] = -m * nu[1]; C_rb[3, 4] = Iz * nu[5]; C_rb[3, 5] = -Iy * nu[4]
        C_rb[4, 0] = -m * nu[2];  C_rb[4, 2] =  m * nu[0]; C_rb[4, 3] = -Iz * nu[5]; C_rb[4, 5] = Ix * nu[3]
        C_rb[5, 0] =  m * nu[1];  C_rb[5, 1] = -m * nu[0]; C_rb[5, 3] = Iy * nu[4]; C_rb[5, 4] = -Ix * nu[3]

        C_a = np.zeros((6, 6))
        # (Simplified C_a filling for brevity, assuming same logic as original file)
        C_a[0, 4] = -self.p.Z_wd * nu[2]; C_a[0, 5] =  self.p.Y_vd * nu[1]
        # ... (Include all other C_a terms from original utils.py) ...
        
        return C_rb + C_a

    def get_x_dot(self, x_state, u_control, disturbance=None):
        """ Calculates x_dot = f(x, u) using Numpy. """
        if disturbance is None: disturbance = np.zeros(6)
        eta = x_state[0:6]
        nu  = x_state[6:12]
        phi, theta, psi = eta[3], eta[4], eta[5]

        tau = self.p.TAM @ u_control
        damping = (self.p.D_LIN @ nu) + (np.abs(nu) * self.p.D_QUAD_COEFFS * nu)
        
        diff = self.p.W - self.p.B
        g_force = np.array([
            -diff * np.sin(theta),
            diff * np.cos(theta) * np.sin(phi),
            diff * np.cos(theta) * np.cos(phi),
            self.p.zg * self.p.W * np.cos(theta) * np.sin(phi),
            self.p.zg * self.p.W * np.sin(theta),
            0.0
        ])

        acc = self.p.M_INV @ (tau - damping - g_force - (self.get_C_np(nu) @ nu) + disturbance)
        
        pos_dot = get_J1_np(phi, theta, psi) @ nu[0:3]
        att_dot = get_J2_np(phi, theta, psi) @ nu[3:6]
        
        return np.concatenate((pos_dot, att_dot, acc))

    def robot_plant_step_RK4(self, x_current, u_control, dt, disturbance=None):
        """ Runge-Kutta 4 Integrator """
        k1 = self.get_x_dot(x_current, u_control, disturbance)
        k2 = self.get_x_dot(x_current + 0.5 * dt * k1, u_control, disturbance)
        k3 = self.get_x_dot(x_current + 0.5 * dt * k2, u_control, disturbance)
        k4 = self.get_x_dot(x_current + dt * k3, u_control, disturbance)
        return x_current + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def generate_target_trajectory(self, steps, dt, speed):
        """ Generates random target movement for testing. """
        # [Copy the exact implementation from your original utils.py here]
        # ... (Logic for random walk/trajectory) ...
        states = np.zeros((steps, 12))
        states[0,0] = 0
        states[0,1] = -6
        states[0,2] = -0.5
        states[0,5] = 0.36
        
        current_x, current_y, current_z = states[0, 0], states[0, 1], states[0, 2]
        current_psi, current_theta = states[0, 5], states[0, 4]

        for k in range(steps - 1):
            target_r = -0.25*np.sin(0.004*k) + 0.1*np.cos(0.003*k)
            target_q = np.random.normal(0.0, 0.1) - (current_theta * 0.1)
            target_u = speed + np.random.normal(0.0, 0.8)
            
            states[k+1, 6] = target_u
            states[k+1, 10] = target_q
            states[k+1, 11] = target_r
            
            current_psi += target_r * dt
            current_theta += target_q * dt
            
            states[k+1, 4] = current_theta
            states[k+1, 5] = current_psi
            
            dx = target_u * np.cos(current_theta) * np.cos(current_psi)
            dy = target_u * np.cos(current_theta) * np.sin(current_psi)
            dz = -target_u * np.sin(current_theta)
            
            current_x += dx * dt
            current_y += dy * dt
            current_z += dz * dt
            
            states[k+1, 0] = current_x
            states[k+1, 1] = current_y
            states[k+1, 2] = current_z
            
        return states

    def get_linear_traj(self, steps, dt, speed):
        """ Generates linear trajectory. """
        # [Copy exact implementation from original utils.py]
        states = np.zeros((steps, 12))
        states[0,0] = 1.8
        states[0,1] = -3
        states[0,2] = -1.5
        states[0,6] = speed
        # ... rest of linear traj logic ...
        return states

    def get_error_avg_std(self, state_estimate, target_state, ref_state):
        """ Calculates performance metrics. """
        # [Copy exact implementation from original utils.py]
        est = np.array(state_estimate)
        tgt = np.array(target_state)
        # ... calculation logic ...
        return {} # dictionary results