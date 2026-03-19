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

        acc = self.p.M_INV @ (tau - damping - g_force + (self.get_C_np(nu) @ nu) + disturbance)
        
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
        states[0,2] = -0.5
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
        states[0,2] = -0.5
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
                states[0,5] += np.deg2rad(np.random.randint(-90,90))
                speed = min(speed_cap, speed + incremental_speed)+0.1
                dx = speed * np.cos(states[0,4]) * np.cos(states[0,5])
                dy = speed * np.cos(states[0,4]) * np.sin(states[0,5])

            speed = max(0,speed - incremental_speed*(2/(min_secs+ max_secs))) 
            states[k+1, 0] = current_x
            states[k+1, 1] = current_y
            states[k+1, 2:11] = states[0, 2:11]

        return states
    def get_spiral_traj(self, steps, dt, speed, is_boat=False):
        """ 
        Generates a trajectory: straight, then 3 large circles (spiraling down if ROV), then straight.
        """
        states = np.zeros((steps, 12))

        # Initial states (matching the linear trajectory starting position)
        states[0,0] = 1.8
        states[0,1] = -3.0
        states[0,2] = 0.0 if is_boat else -2.0
        states[0,4] = 0.0  # Pitch
        states[0,5] = 0.0  # Yaw
        states[0,6] = speed

        current_x, current_y, current_z = states[0, 0], states[0, 1], states[0, 2]
        current_theta, current_psi = states[0, 4], states[0, 5]

        # Phase 1: Straight line for the first few seconds
        t_straight = 10.0
        k_straight = int(t_straight / dt)

        # Phase 2: Big circles
        target_radius = 5.0
        yaw_rate = speed / target_radius 
        
        # Time required to complete exactly 3 circles (3 * 2pi radians)
        t_circles = (3 * 2 * np.pi) / yaw_rate
        k_circles = int(t_circles / dt)

        # Depth parameters: dive by 6 meters (ending at -8.0m for the ROV)
        z_drop_total = -6.0 
        z_rate = z_drop_total / t_circles if not is_boat else 0.0

        for k in range(steps - 1):
            target_u = speed
            target_r = 0.0
            
            if k < k_straight:
                # PHASE 1: Straight
                target_r = 0.0
                current_theta = 0.0
            elif k < k_straight + k_circles:
                # PHASE 2: 3 Circles and Dive
                target_r = yaw_rate
                if not is_boat:
                    # Calculate required pitch to achieve the dive rate
                    # Since dz = -u * sin(theta), we invert this to find the required pitch angle
                    req_theta = np.arcsin(np.clip(-z_rate / speed, -1.0, 1.0))
                    current_theta = req_theta
            else:
                # PHASE 3: Straight again
                target_r = 0.0
                current_theta = 0.0 # Level out to hold the new depth

            # Integrate orientations
            current_psi += target_r * dt

            # Record states
            states[k+1, 6] = target_u
            states[k+1, 10] = 0.0 
            states[k+1, 11] = target_r
            states[k+1, 4] = current_theta
            states[k+1, 5] = current_psi

            # Kinematic position update
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

    def get_mixed_traj(self, steps, dt, speed, is_boat=False):
        """ 
        Ultimate Gauntlet: 
        Phase 1: Straight for 10m 
        Phase 2: 1 Full Circle (Diving if ROV)
        Phase 3: Sharp angles (ankle breakers) and aggressive accel/decel.
        """
        states = np.zeros((steps, 12))

        # Initial states
        states[0,0] = 1.8
        states[0,1] = -3.0
        states[0,2] = 0.0 if is_boat else -0.5
        states[0,4] = 0.0  # Pitch
        states[0,5] = 0.0  # Yaw
        states[0,6] = speed

        current_x, current_y, current_z = states[0, 0], states[0, 1], states[0, 2]
        current_theta, current_psi = states[0, 4], states[0, 5]

        # Tracking variables for phase transitions
        dist_covered = 0.0
        phase = 1
        circle_steps = 0
        
        # Phase 2 Parameters
        target_radius = 5.0
        yaw_rate = speed / target_radius 
        k_circle_total = int((2 * np.pi) / yaw_rate / dt)
        
        z_drop_total = -6.0 
        z_rate = z_drop_total / (k_circle_total * dt) if not is_boat else 0.0

        # Phase 3 Parameters (from get_random_traj)
        next_turn_step = 0
        min_secs = 2
        max_secs = 11
        incremental_speed = 0.7
        speed_cap = 1.1

        for k in range(steps - 1):
            target_r = 0.0
            target_q = 0.0
            
            # --- PHASE LOGIC ---
            if phase == 1:
                # PHASE 1: Straight until 10 meters covered
                dist_covered += speed * dt
                if dist_covered >= 10.0:
                    phase = 2
                    
            elif phase == 2:
                # PHASE 2: One Circle and Dive
                target_r = yaw_rate
                if not is_boat:
                    req_theta = np.arcsin(np.clip(-z_rate / speed, -1.0, 1.0))
                    current_theta = req_theta
                
                circle_steps += 1
                if circle_steps >= k_circle_total:
                    phase = 3
                    current_theta = 0.0 # Level out for the random walk
                    # Schedule the first random sharp turn
                    next_turn_step = k + int(np.random.randint(min_secs, max_secs) / dt)
                    
            elif phase == 3:
                # PHASE 3: Sharp Angles & Accelerations
                if k >= next_turn_step:
                    # The "Ankle Breaker": Instant heading change (-90 to +90 degrees)
                    current_psi += np.deg2rad(np.random.randint(-90, 90))
                    
                    # Aggressive acceleration spike
                    speed = min(speed_cap, speed + incremental_speed) + 0.1
                    
                    # Schedule the next turn
                    next_turn_step = k + int(np.random.randint(min_secs, max_secs) / dt)

                # Gradual deceleration between turns to force the controller to manage speed changes
                speed = max(0.2, speed - incremental_speed * (2 / (min_secs + max_secs)) * dt)

            # --- KINEMATICS INTEGRATION ---
            current_psi += target_r * dt
            current_theta += target_q * dt

            # Record states
            states[k+1, 6] = speed
            states[k+1, 10] = target_q 
            states[k+1, 11] = target_r
            states[k+1, 4] = current_theta
            states[k+1, 5] = current_psi

            # Position update
            dx = speed * np.cos(current_theta) * np.cos(current_psi)
            dy = speed * np.cos(current_theta) * np.sin(current_psi)
            dz = -speed * np.sin(current_theta)

            current_x += dx * dt
            current_y += dy * dt
            current_z += dz * dt

            # Hard lock the boat to the surface
            if is_boat:
                current_z = 0.0

            states[k+1, 0] = current_x
            states[k+1, 1] = current_y
            states[k+1, 2] = current_z

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
    
    def force_w2b(self, rov_state, force_world):
        """
        Rotate Force into Body frame
        """
        phi, theta, psi = rov_state[3],rov_state[4], rov_state[5]
        R_B2W = self.get_J1_np(phi, theta, psi)

        force_lin_body = R_B2W.T @ force_world[0:3]

        return force_lin_body
    
    def get_J1_np(self, phi, theta, psi):
        """
        Numpy implementation of Rotation Matrix (Body -> World).
        Standard Z-Y-X rotation sequence.
        """
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        r11 = cpsi * cth
        r12 = -spsi * cphi + sphi * sth * cpsi
        r13 = spsi * sphi + sth * cpsi * cphi

        r21 = spsi * cth
        r22 = cpsi * cphi + sphi * sth * spsi
        r23 = -cpsi * sphi + sth * spsi * cphi

        r31 = -sth
        r32 = sphi * cth
        r33 = cphi * cth

        J1 = np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])
        return J1