import numpy as np
import casadi as cas
import matplotlib.pyplot as plt

class Vehicle_Utils:
    def __init__(self, params):
        self.p = params

    def get_shadow_ref(self, rov_state, target_state, target_vel = None, desired_dist = 2.0):
        """
        Calculates the 'Ideal State' (Shadow) for the NMPC to track.
        Not using Finite derivative as amplifies noise
        Not using LPF as introduces lag: takes 0.5sec to realize the movement
        Not using Particle Filter as is an "overkill" 
        """
        p_rov = rov_state[0:3]
        psi_rov = rov_state[5]

        p_target = target_state[0:3]

        v_target_global = target_vel if target_vel is not None else np.zeros(3)

        error_vector = p_rov - p_target
        dist_3d = np.linalg.norm(error_vector)

        if dist_3d < 0.01:
            direction_vect = np.array([-1.0, 0.0, 0.0])  # Default direction
        else:
            direction_vect = error_vector / dist_3d

        p_reference = p_target + direction_vect * desired_dist


        v_ref_global = v_target_global
        
        # Defining orientation with ROV facing the target:
        target_pointing_vector = p_target - p_reference
        yaw_des = np.arctan2(target_pointing_vector[1], target_pointing_vector[0])

        diff = yaw_des - rov_state[5]
        if abs(diff) > np.pi:
            # If error is big due to wrapping, shift the ref to match the ROV i think
            if diff > np.pi:
                yaw_des -= 2 * np.pi
            elif diff < -np.pi:
                yaw_des += 2 * np.pi

        dist_plane = np.linalg.norm(target_pointing_vector[0:2])
        pitch_des = np.arctan2(-target_pointing_vector[2], dist_plane)

        c_psi = np.cos(yaw_des)
        s_psi = np.sin(yaw_des)
        u_ref = v_ref_global[0] * c_psi + v_ref_global[1] * s_psi
        v_ref = -v_ref_global[0] * s_psi + v_ref_global[1] * c_psi
        w_ref = v_ref_global[2]
        

        ref_state = np.zeros(12)
        ref_state[0] = p_reference[0]
        ref_state[1] = p_reference[1]
        ref_state[2] = p_reference[2]
        ref_state[3] = 0  # Roll 0
        ref_state[4] = pitch_des
        ref_state[5] = yaw_des  # DESIRED HEADING
        ref_state[6] = u_ref   # DESIRED SURGE
        ref_state[7] = v_ref   # DESIRED SWAY
        ref_state[8] = w_ref   # Desired Heave 

        return ref_state

    def get_shadow_traj(self, rov_state, target_state, target_vel, dt, horizon_N, desired_dist = 2.0):
        """
        Generates a list of N reference states (The Trajectory) for the NMPC.
        """
        ref_state_t0 = self.get_shadow_ref(rov_state, target_state, target_vel, desired_dist)
        
        
        p_ref_current = ref_state_t0[0:3]
        v_ref_current = np.array([ref_state_t0[6], ref_state_t0[7], ref_state_t0[8]])
        v_global_propagation = target_vel if target_vel is not None else np.zeros(3)

        trajectory = []

        # Loop to generate the future points
        for k in range(int(horizon_N/5)):
            
            time_offset = k * dt
            
            
            # We assume the target moves at constant velocity, so the shadow does too.
            p_ref_future = p_ref_current + v_global_propagation * time_offset
            
            ref_state_k = ref_state_t0.copy()
            ref_state_k[0:3] = p_ref_future
            trajectory.append(ref_state_k)

        return np.array(trajectory) # Shape (N, 12)



    def get_J1(self, phi, theta, psi):
        """ Rotation Matrix (Body -> World) using CasADi SX """
        cphi, sphi = cas.cos(phi), cas.sin(phi)
        cth, sth  = cas.cos(theta), cas.sin(theta)
        cpsi, spsi = cas.cos(psi), cas.sin(psi)

        return cas.vertcat(
            cas.horzcat(cpsi*cth, -spsi*cphi + sphi*sth*cpsi, spsi*sphi + sth*cpsi*cphi),
            cas.horzcat(spsi*cth,  cpsi*cphi + sphi*sth*spsi, -cpsi*sphi + sth*spsi*cphi),
            cas.horzcat(-sth,      sphi*cth,                  cphi*cth)
        )

    def get_J2(self, phi, theta, psi):
        ttheta = cas.tan(theta)
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth = np.cos(theta)
        J2 = cas.vertcat(
            cas.horzcat(1, sphi*ttheta, cphi*ttheta),
            cas.horzcat(0, cphi, -sphi),
            cas.horzcat(0, sphi/cth, cphi/cth)
        )
        return J2

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

    def get_J2_np(self, phi, theta, psi):
        """
        Numpy implementation of Angular Velocity Transformation.
        Maps Body rates [p, q, r] to Euler rates [dphi, dtheta, dpsi].
        """
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth = np.cos(theta)
        ttheta = np.tan(theta)

        # Prevent division by zero singularity at pitch = +/- 90 deg
        if abs(cth) < 0.01: 
            cth = np.sign(cth) * 0.01

        J2 = np.array([
            [1, sphi * ttheta, cphi * ttheta],
            [0, cphi,         -sphi],
            [0, sphi / cth,    cphi / cth]
        ])
        return J2
    # def get_C_cas(self, nu):
    #     C_rb = cas.MX.zeros(6, 6)
    #     # Row 0
    #     C_rb[0, 4] =  self.p.m * nu[2];  C_rb[0, 5] = -self.p.m * nu[1]
    #     # Row 1
    #     C_rb[1, 3] = -self.p.m * nu[2];  C_rb[1, 5] =  self.p.m * nu[0]
    #     # Row 2
    #     C_rb[2, 3] =  self.p.m * nu[1];  C_rb[2, 4] = -self.p.m * nu[0]
    #     C_rb[3, 1] =  self.p.m * nu[2];  C_rb[3, 2] = -self.p.m * nu[1]; C_rb[3, 4] = self.p.Iz * nu[5]; C_rb[3, 5] = -self.p.Iy * nu[4]
    #     C_rb[4, 0] = -self.p.m * nu[2];  C_rb[4, 2] =  self.p.m * nu[0]; C_rb[4, 3] = -self.p.Iz * nu[5]; C_rb[4, 5] = self.p.Ix * nu[3]
    #     C_rb[5, 0] =  self.p.m * nu[1];  C_rb[5, 1] = -self.p.m * nu[0]; C_rb[5, 3] = self.p.Iy * nu[4]; C_rb[5, 4] = -self.p.Ix * nu[3]

    #     C_a = cas.MX.zeros(6, 6)
    #     C_a[0, 4] = -self.p.Z_wd * nu[2]; C_a[0, 5] =  self.p.Y_vd * nu[1]
    #     C_a[1, 3] =  self.p.Z_wd * nu[2]; C_a[1, 5] = -self.p.X_ud * nu[0]
    #     C_a[2, 3] = -self.p.Y_vd * nu[1]; C_a[2, 4] =  self.p.X_ud * nu[0]
    #     C_a[3, 1] = -self.p.Z_wd * nu[2]; C_a[3, 2] =  self.p.Y_vd * nu[1]; C_a[3, 4] = -self.p.N_rd * nu[5]; C_a[3, 5] = self.p.M_qd * nu[4]
    #     C_a[4, 0] =  self.p.Z_wd * nu[2]; C_a[4, 2] = -self.p.X_ud * nu[0]; C_a[4, 3] =  self.p.N_rd * nu[5]; C_a[4, 5] = -self.p.K_pd * nu[3]
    #     C_a[5, 0] = -self.p.Y_vd * nu[1]; C_a[5, 1] =  self.p.X_ud * nu[0]; C_a[5, 3] = -self.p.M_qd * nu[4]; C_a[5, 4] = self.p.K_pd * nu[3]

    #     coriolis_sum = C_rb 

    #     return coriolis_sum
    # def get_C_SX(self, nu):
    #     """
    #     Computes Coriolis matrix using strictly CasADi SX types 
    #     to match the model definition.
    #     """
    #     # 1. Create Empty SX Matrix (NOT np.zeros, NOT MX.zeros)
    #     C_rb = cas.SX.zeros(6, 6)
        
    #     m = self.p.m
    #     # Use explicit float casting for constants to be safe
    #     Ix, Iy, Iz = float(self.p.Ix), float(self.p.Iy), float(self.p.Iz)
        
    #     # 2. Fill Rigid Body Coriolis (Standard SNAME)
    #     # Note: We access nu by index directly
    #     u, v, w = nu[0], nu[1], nu[2]
    #     p, q, r = nu[3], nu[4], nu[5]

    #     # Row 0
    #     C_rb[0, 4] =  m * w;   C_rb[0, 5] = -m * v
    #     # Row 1
    #     C_rb[1, 3] = -m * w;   C_rb[1, 5] =  m * u
    #     # Row 2
    #     C_rb[2, 3] =  m * v;   C_rb[2, 4] = -m * u
    #     # Row 3
    #     C_rb[3, 1] = -m * w;   C_rb[3, 2] =  m * v;   C_rb[3, 4] =  Iz * r;   C_rb[3, 5] = -Iy * q
    #     # Row 4
    #     C_rb[4, 0] =  m * w;   C_rb[4, 2] = -m * u;   C_rb[4, 3] = -Iz * r;   C_rb[4, 5] =  Ix * p
    #     # Row 5
    #     C_rb[5, 0] = -m * v;   C_rb[5, 1] =  m * u;   C_rb[5, 3] =  Iy * q;   C_rb[5, 4] = -Ix * p
        
    #     # 3. Added Mass Coriolis (C_a)
    #     C_a = cas.SX.zeros(6, 6)
    #     Xud, Yvd, Zwd = float(self.p.X_ud), float(self.p.Y_vd), float(self.p.Z_wd)
    #     Kpd, Mqd, Nrd = float(self.p.K_pd), float(self.p.M_qd), float(self.p.N_rd)

    #     # Approximated diagonal Added Mass Coriolis
    #     # (Check your specific notation signs, these follow Fossen generally)
    #     a1, a2, a3 = Xud*u, Yvd*v, Zwd*w
    #     b1, b2, b3 = Kpd*p, Mqd*q, Nrd*r

    #     C_a[0, 5] = -a2;   C_a[0, 4] =  a3
    #     C_a[1, 5] =  a1;   C_a[1, 3] = -a3
    #     C_a[2, 4] = -a1;   C_a[2, 3] =  a2

    #     C_a[3, 5] = -b2;   C_a[3, 4] =  b3;  C_a[3, 2] = -a2;  C_a[3, 1] =  a3
    #     C_a[4, 5] = -b1;   C_a[4, 3] = -b3;  C_a[4, 2] =  a1;  C_a[4, 0] = -a3
    #     C_a[5, 4] =  b1;   C_a[5, 3] =  b2;  C_a[5, 1] = -a1;  C_a[5, 0] =  a2

    #     # Note: C_np had some sign diffs and transposition relative to standard Fossen.
    #     # Ensure this matches your specific model notation (SNAME vs Fossen 1994).
        
    #     return C_rb + C_a

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

    def get_x_dot(self, x_state, u_control, disturbance = None):
        """
        Calculates x_dot = f(x, u) using purely Numpy math.
        """
        if disturbance is None:
            disturbance = np.zeros(6)
        eta = x_state[0:6]
        nu  = x_state[6:12]
        
        phi, theta, psi = eta[3], eta[4], eta[5]

        tau = self.p.TAM @ u_control
        damping_lin = self.p.D_LIN @ nu
        dq_diag = np.abs(nu) * self.p.D_QUAD_COEFFS 
        damping_quad = dq_diag * nu 
        W, B, zg = self.p.W, self.p.B, self.p.zg
        diff = W - B
        
        g_force = np.array([
            -diff * np.sin(theta),
            diff * np.cos(theta) * np.sin(phi),
            diff * np.cos(theta) * np.cos(phi),
            zg * W * np.cos(theta) * np.sin(phi),
            zg * W * np.sin(theta),
            0.0
        ])

        # Coriolis???
        C_mat = self.get_C_np(nu) 
        coriolis_force = C_mat @ nu
        total_force = tau - damping_lin - damping_quad - g_force - coriolis_force + disturbance
        acc = self.p.M_INV @ total_force

        J1 = self.get_J1_np(phi, theta, psi)
        J2 = self.get_J2_np(phi, theta, psi)
        pos_dot = J1 @ nu[0:3]
        att_dot = J2 @ nu[3:6]
        
        return np.concatenate((pos_dot, att_dot, acc))

    def robot_plant_step_RK4(self, x_current, u_control, dt, disturbance = None):
        """
        Simulates one time step using Runge-Kutta 4 (RK4).
        Much more stable than Euler for Coriolis forces.
        """
        if disturbance is None:
            disturbance = np.zeros(6)

        # RK4 here
        k1 = self.get_x_dot(x_current, u_control, disturbance)
        k2 = self.get_x_dot(x_current + 0.5 * dt * k1, u_control, disturbance)
        k3 = self.get_x_dot(x_current + 0.5 * dt * k2, u_control, disturbance)
        k4 = self.get_x_dot(x_current + dt * k3, u_control, disturbance)
        
        x_next = x_current + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return x_next

    def robot_plant_step(self, x_current, u_control, dt):
        """
        Simulates the robot moving for one time step using NumPy.
        """
        eta = x_current[0:6]
        nu  = x_current[6:12]
    
        tau = self.p.TAM @ u_control
        damping_force = self.p.D_LIN @ nu
        W = self.p.W
        B = self.p.B
        zg = self.p.zg
        diff = W - B
        
        g_force = np.array([
            -diff * cas.sin(x_current[5]),
            diff * cas.cos(x_current[5]) * cas.sin(x_current[4]),
            diff * cas.cos(x_current[5]) * cas.cos(x_current[4]),
            zg * W * cas.cos(x_current[5]) * cas.sin(x_current[4]),
            zg * W * cas.sin(x_current[5]),
            0
        ])
        coriolis_sum = self.get_C_np(nu)
        coriolis_force = coriolis_sum @ nu
        total_force = tau - damping_force - g_force - coriolis_force
        acc = self.p.M_INV @ total_force

        J1 = self.get_J1(eta[3], eta[4], eta[5])
        J2 = self.get_J2(eta[3], eta[4], eta[5])
        pos_dot = J1 @ nu[0:3]
        att_dot = J2 @nu[3:6] 

        # Integration
        x_next = np.zeros(12)
        x_next[0:3] = np.squeeze(eta[0:3] + pos_dot[0:3] * dt)
        x_next[3:6] = np.squeeze(eta[3:6] + att_dot[0:3] * dt)
        x_next[6:12] = np.squeeze(nu + acc * dt)
        
        return x_next

    def get_linear_traj(self, steps, dt, speed):
        """
        Generates a linear trajectory from start to end in given steps.
        """
        states = np.zeros((steps, 12))
        
        states[0,0] = 1.8
        states[0,1] = -3
        states[0,2] = -1.5

        states[0,4] = 0.5  # Pitch
        states[0,5] = 0.0  # Yaw
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

    def generate_target_trajectory(self, steps, dt, speed):
        """
        Generates a 12-state trajectory for MPC testing.
        Returns: numpy array of shape (steps, 12)
        States: [x, y, z, phi, theta, psi, u, v, w, p, q, r]
        """
        states = np.zeros((steps, 12))
        
        states[0,0] = 0
        states[0,1] = -6
        states[0,2] = -0.5

        states[0,4] = 0.5  # Pitch
        #states[0,4] = 0.0  # Pitch

        states[0,5] = 0.36  # Yaw

        # Simulation Variables
        current_x, current_y, current_z = states[0, 0], states[0, 1], states[0, 2]
        current_psi = states[0, 5]  # Yaw
        current_theta = states[0, 4] # Pitch

        for k in range(steps - 1):
            # rnd Yaw (r) L/R
            target_r = -0.1*np.sin(0.004*k) + 0.08*np.sin(0.0003*k)  #turning L/R
            target_r = -0.25*np.sin(0.004*k) + 0.1*np.cos(0.003*k)  #crazier traj


            # rnd Pitch (q) Up/DOwn
            target_q = np.random.normal(0.0, 0.1) - (current_theta * 0.1)
            #target_q = 0
            
            
            # Roll (p) is usually 0 for a stable target
            target_p = 0.0
            
            target_u = speed + np.random.normal(0.0, 0.8)
            target_v = 0.0
            target_w = 0.0

            states[k+1, 6] = target_u
            states[k+1, 7] = target_v
            states[k+1, 8] = target_w
            states[k+1, 9] = target_p
            states[k+1, 10] = target_q
            states[k+1, 11] = target_r

            current_psi += target_r * dt
            current_theta += target_q * dt
            
            states[k+1, 3] = 0.0 # Roll (phi)
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
    