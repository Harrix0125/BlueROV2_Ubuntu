import numpy as np
from nmpc_params import BlueROV_Params as MPCC
class VisualTarget:
    def __init__(self, start_state, fov_h = 90, fov_v = 80, max_dist = 10):
        self.true_state = np.array(start_state[0:6])

        #seen_state: last position measured by camera (with noise)
        self.seen_state = np.zeros(3)
        self.estimated_state = np.zeros(6)

        self.kf = TargetTrackerKF(dt_default=MPCC.T_s)

        self.fov_h_rad = np.deg2rad(fov_h)
        self.fov_v_rad = np.deg2rad(fov_v)
        self.max_dist = max_dist

        self.is_visible = True
        self.last_seen_t = 0

    def check_visibility(self, rov_state, seen_it_once = True):
        """
        Check if the target is within the ROV's camera FOV and range.
        rov_state: [x, y, z, phi, theta, psi, u, v, w, p, q, r]
        """
        dx = self.true_state[0] - rov_state[0]
        dy = self.true_state[1] - rov_state[1]
        dz = self.true_state[2] - rov_state[2]

        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        if distance > self.max_dist:
            self.is_visible = False
            return self.is_visible

        # Calculate angles in ROV frame
        yaw = rov_state[5]
        x_rel =  np.cos(-yaw) * dx - np.sin(-yaw) * dy
        y_rel =  np.sin(-yaw) * dx + np.cos(-yaw) * dy
        z_rel = dz

        angle_h = np.arctan2(y_rel, x_rel)
        angle_v = np.arctan2(z_rel, np.sqrt(x_rel**2 + y_rel**2))

        if (abs(angle_h) <= self.fov_h_rad / 2) and (abs(angle_v) <= self.fov_v_rad / 2):
            self.is_visible = True
            if seen_it_once == False:
                #   Setting first time we see it the state like this so that it wil trust more the physics, else we start it at 0-0-0
                self.kf.set_state(self.true_state)
        else:
            self.is_visible = False

        return self.is_visible

    def get_camera_estimate(self, rov_state, dt = MPCC.T_s, camera_noise = np.array([0.0, 0.0, 0.0])):
        """
        Update the target state estimate based on visibility.
        If visible, perform KF update with measurement.
        If not visible, only predict.
        rov_state: [x, y, z, phi, theta, psi, u, v, w, p, q, r]
        """
        self.kf.predict(dt)

        dx = self.true_state[0] - rov_state[0]
        dy = self.true_state[1] - rov_state[1]
        dz = self.true_state[2] - rov_state[2]
        distance = np.sqrt(dx**2 + dy**2 + dz**2)

        if self.check_visibility(rov_state):
            mult_noise = distance ** (np.log(10) / np.log(2.2))
            self.seen_state = self.true_state[0:3] + camera_noise*mult_noise
            measurement = self.true_state[0:3]
            self.kf.update(self.seen_state)
            self.last_seen_t = 0
            
        else:
            measurement = self.kf.get_state()[0:3]  # No new measurement, use prediction
            self.kf.update(measurement)
            self.last_seen_t += dt
        
        self.estimated_state = self.kf.get_state()
        return self.estimated_state

    def truth_update(self, state_moving):
        self.true_state = np.array(state_moving[0:6])

    def get_time_since_seen(self):
        return self.last_seen_t

        



class TargetTrackerKF:
    def __init__(self, dt_default=0.1, process_noise=0.05, measure_noise=0.5):
        """
            K filter for 3D position + velocity tracking.
            state vector x :[x, y, z, vx, vy, vz]
        """
        self.dt = dt_default
        self.n_states = 6  # [x, y, z, vx, vy, vz]
        self.n_meas = 3  # [x, y, z]

        # State Vector
        self.x_est = np.zeros(self.n_states)

        # Covariance Matrix
        self.P_est = np.eye(self.n_states) * 1.0  # Initial 

        # State Transition Matrix
        self.F = np.eye(self.n_states)
        self.F[0,3] = dt_default
        self.F[1,4] = dt_default
        self.F[2,5] = dt_default

        # Measurement Matrix
        self.H = np.zeros((self.n_meas, self.n_states))
        self.H[0,0] = 1
        self.H[1,1] = 1
        self.H[2,2] = 1

        # Process Noise Covariance
        self.Q = np.eye(self.n_states) * process_noise

        # Measurement Noise Covariance
        self.R = np.eye(self.n_meas)* measure_noise

        self.I = np.eye(self.n_states)

    def predict(self, dt = None):
        """
        Predict step: estimating where the target IS right now based on previous velocity.
        Useful if camera frames drop or lag.
        """
        if dt is not None and dt != self.dt:
            self.F[0,3] = dt
            self.F[1,4] = dt
            self.F[2,5] = dt       
            self.dt = dt

        self.x_est = self.F @ self.x_est
        self.P_est = self.F @ self.P_est @ self.F.T + self.Q

    def update(self, measurement):
        """
        Update step: Correcting the estimate with new camera data.
        measurement: [x, y, z] from camera
        """
        z = np.array(measurement)

        # Residual
        y = z - (self.H @ self.x_est)

        # Uncertainty
        S = self.H @ self.P_est @ self.H.T + self.R

        # Kalman Gain
        try:
            K = self.P_est @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Singular matrix in Kalman Gain calculation.")
            K = np.zeros((self.n_states, self.n_meas))

        # Update State and Covariance
        self.x_est = self.x_est + K @ y 
        self.P_est = (self.I - K @ self.H) @ self.P_est

    def get_state(self):
        return self.x_est
    
    def set_state(self, new_x_est):
        self.x_est = new_x_est