import numpy as np
import casadi as cas
from core.kinematics import get_J1_np, get_J2_np, get_J1, get_J2

class Vehicle_Sim_Utils:
    def __init__(self, params):
        self.p = params

    def get_C_np(self, nu):
        C_rb = np.zeros((6, 6))
        # Row 0
        C_rb[0, 4] =  self.p.m * nu[2];  C_rb[0, 5] = -self.p.m * nu[1]
        # Row 1
        C_rb[1, 3] = -self.p.m * nu[2];  C_rb[1, 5] =  self.p.m * nu[0]
        # Row 2
        C_rb[2, 3] =  self.p.m * nu[1];  C_rb[2, 4] = -self.p.m * nu[0]
        C_rb[3, 1] =  self.p.m * nu[2];  C_rb[3, 2] = -self.p.m * nu[1]; C_rb[3, 4] = self.p.Iz * nu[5]; C_rb[3, 5] = -self.p.Iy * nu[4]
        C_rb[4, 0] = -self.p.m * nu[2];  C_rb[4, 2] =  self.p.m * nu[0]; C_rb[4, 3] = -self.p.Iz * nu[5]; C_rb[4, 5] = self.p.Ix * nu[3]
        C_rb[5, 0] =  self.p.m * nu[1];  C_rb[5, 1] = -self.p.m * nu[0]; C_rb[5, 3] = self.p.Iy * nu[4]; C_rb[5, 4] = -self.p.Ix * nu[3]

        C_a = np.zeros((6, 6))
        C_a[0, 4] = -self.p.Z_wd * nu[2]; C_a[0, 5] =  self.p.Y_vd * nu[1]
        C_a[1, 3] =  self.p.Z_wd * nu[2]; C_a[1, 5] = -self.p.X_ud * nu[0]
        C_a[2, 3] = -self.p.Y_vd * nu[1]; C_a[2, 4] =  self.p.X_ud * nu[0]
        C_a[3, 1] = -self.p.Z_wd * nu[2]; C_a[3, 2] =  self.p.Y_vd * nu[1]; C_a[3, 4] = -self.p.N_rd * nu[5]; C_a[3, 5] = self.p.M_qd * nu[4]
        C_a[4, 0] =  self.p.Z_wd * nu[2]; C_a[4, 2] = -self.p.X_ud * nu[0]; C_a[4, 3] =  self.p.N_rd * nu[5]; C_a[4, 5] = -self.p.K_pd * nu[3]
        C_a[5, 0] = -self.p.Y_vd * nu[1]; C_a[5, 1] =  self.p.X_ud * nu[0]; C_a[5, 3] = -self.p.M_qd * nu[4]; C_a[5, 4] = self.p.K_pd * nu[3]

        coriolis_sum =  C_a + C_rb

        return coriolis_sum

    def get_x_dot(self, x_state, u_control, disturbance=None):
        """ 
        Calculates x_dot = f(x u) using Numpy. 
        """
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
        """ 
        Runge-Kutta 4 Integrator 
        """
        if disturbance is None:
            disturbance = np.zeros(6)

        k1 = self.get_x_dot(x_current, u_control, disturbance)
        k2 = self.get_x_dot(x_current + 0.5 * dt * k1, u_control, disturbance)
        k3 = self.get_x_dot(x_current + 0.5 * dt * k2, u_control, disturbance)
        k4 = self.get_x_dot(x_current + dt * k3, u_control, disturbance)
        return x_current + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def generate_target_trajectory(self, steps, dt, speed):
        """ 
        Generates random target movement for testing. 
        """
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
        states = np.zeros((steps, 12))

        states[0,0] = 1.8
        states[0,1] = -3
        states[0,2] = -1.5
        states[0,6] = speed
        
        current_x, current_y = states[0, 0], states[0, 1]
        dx = speed * np.cos(states[0,4]) * np.cos(states[0,5])
        dy = speed * np.cos(states[0,4]) * np.sin(states[0,5])

        for k in range(steps-1):
            current_x += dx * dt
            current_y += dy * dt
            states[k+1, 0] = current_x
            states[k+1, 1] = current_y
            states[k+1, 2:11] = states[0, 2:11]
        return states
    
    def get_random_traj(self, steps, dt, speed):
        """ Generates rnd trajectory. """
        states = np.zeros((steps, 12))

        states[0,0] = 1.8
        states[0,1] = -3
        states[0,2] = -1.5
        states[0,6] = speed
        change_dir = []
        seconds = steps*dt
        act_steps = 0
        incremental_speed = 0.8
        min_secs = 2
        max_secs = 11
        speed_cap = 1.9
        
        while (act_steps*dt) < seconds:
            ankle_breaker = np.random.randint(min_secs/dt, max_secs/dt)      # in steps
            if (ankle_breaker*dt + act_steps*dt) > seconds:     # sec + sec > sec
                break
            act_steps += ankle_breaker                          # step += steps
            change_dir.append(act_steps)                        # append the step
        current_x, current_y = states[0, 0], states[0, 1]
        dx = speed * np.cos(states[0,4]) * np.cos(states[0,5])
        dy = speed * np.cos(states[0,4]) * np.sin(states[0,5])
        number_ankles_broken = 0   # index of #change directions
        
        for k in range(steps-1):
            current_x += dx * dt
            current_y += dy * dt

            if number_ankles_broken < len(change_dir) and k == change_dir[number_ankles_broken]:
                number_ankles_broken +=1
                states[0,5] += np.deg2rad(np.random.randint(-75,75))
                speed = min(speed_cap, speed + incremental_speed)+0.1
                dx = speed * np.cos(states[0,4]) * np.cos(states[0,5])
                dy = speed * np.cos(states[0,4]) * np.sin(states[0,5])

            speed = max(0,speed - incremental_speed*(2/(min_secs+ max_secs))) 
            states[k+1, 0] = current_x
            states[k+1, 1] = current_y
            states[k+1, 2:11] = states[0, 2:11]

        return states

    def get_error_avg_std(self, state_estimate, target_state, ref_state):
        """
        Calculates Mean Absolute Error (MAE) and Euclidean distance.
        Arguments must be convertible to numpy arrays of shape (3, N).
        """
        # Convert everything to consistent NumPy arrays (3, N)
        #    Rows = x, y, z; Cols = Time steps
        est = np.array(state_estimate) 
        tgt = np.array(target_state)
        ref = np.array(ref_state)

        # Safety check for shapes
        if est.shape != tgt.shape:
            # Handle case where target might be transposed compared to estimate
            if est.shape == tgt.T.shape:
                tgt = tgt.T
            elif est.shape == tgt.shape:
                tgt = tgt
            else:
                min_len = min(est.shape[1], tgt.shape[1])
                min_len = min(min_len, ref.shape[1])
                est = est[:, :min_len]
                tgt = tgt[:, :min_len]
                print(f"Synchronized to length: {min_len}")
                print(f"Shape mismatch! Est: {est.shape}, Tgt: {tgt.shape}")
                ref = ref[:,:min_len]

        # Calculate Errors for TARGET
        # Difference at every step
        diff_matrix = est - tgt
        
        # Per-axis Mean Absolute Error (MAE)
        # Average across time (axis 1)
        mae_x, mae_y, mae_z = np.mean(np.abs(diff_matrix), axis=1)
        
        # 3D Euclidean Distance (Average tracking error)
        # Norm at each step, then mean
        dist_3d = np.linalg.norm(diff_matrix, axis=0)
        avg_3d_dist = np.mean(dist_3d)

        # Calculate Errors for REFERENCE (Virtual Carrot)
        diff_ref = est - ref
        dist_ref_3d = np.linalg.norm(diff_ref, axis=0)
        avg_ref_dist = np.mean(dist_ref_3d)

        print("-" * 30)
        print(f"TRACKING PERFORMANCE:")
        print(f"  Avg 3D Error: {avg_3d_dist:.4f} m")
        print(f"  MAE X: {mae_x:.4f} m")
        print(f"  MAE Y: {mae_y:.4f} m")
        print(f"  MAE Z: {mae_z:.4f} m")
        print(f"  Avg Dist from Ref: {avg_ref_dist:.4f} m")
        print("-" * 30)

        return {
            'mae_x': mae_x,
            'mae_y': mae_y,
            'mae_z': mae_z,
            'avg_3d': avg_3d_dist
        }