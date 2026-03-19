import numpy as np
from config.nmpc_params import GazeboROV_Params as MPCC

class VisualTarget:
    def __init__(self, start_state, fov_h=90, fov_v=80, max_dist=10):
        self.true_state = np.array(start_state[0:6])

        # seen_state: last position measured by camera (with noise)
        self.seen_state = np.zeros(3)
        self.estimated_state = np.zeros(6)

        # Replaced the single KF with the IMM Tracker
        self.tracker = IMMTracker(dt_default=MPCC.T_s)

        self.fov_h_rad = np.deg2rad(fov_h)
        self.fov_v_rad = np.deg2rad(fov_v)
        self.max_dist = max_dist

        self.is_visible = True
        self.last_seen_t = 0

    def check_visibility(self, rov_state, seen_it_once=True):
        """
        Check if the target is within the ROV's camera FOV and range.
        """
        dx = self.true_state[0] - rov_state[0]
        dy = self.true_state[1] - rov_state[1]
        dz = self.true_state[2] - rov_state[2]

        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        if distance > self.max_dist:
            self.is_visible = False
            return self.is_visible

        yaw = rov_state[5]
        x_rel =  np.cos(-yaw) * dx - np.sin(-yaw) * dy
        y_rel =  np.sin(-yaw) * dx + np.cos(-yaw) * dy
        z_rel = dz

        angle_h = np.arctan2(y_rel, x_rel)
        angle_v = np.arctan2(z_rel, np.sqrt(x_rel**2 + y_rel**2))

        if (abs(angle_h) <= self.fov_h_rad / 2) and (abs(angle_v) <= self.fov_v_rad / 2):
            self.is_visible = True
            if not seen_it_once:
                self.tracker.set_state(self.true_state)
        else:
            self.is_visible = False

        return self.is_visible

    def get_camera_estimate(self, rov_state, dt=MPCC.T_s, camera_noise=np.array([0.0, 0.0, 0.0])):
        """
        Update the target state estimate based on visibility using IMM.
        """
        self.tracker.predict(dt)

        dx = self.true_state[0] - rov_state[0]
        dy = self.true_state[1] - rov_state[1]
        dz = self.true_state[2] - rov_state[2]
        distance = np.sqrt(dx**2 + dy**2 + dz**2)

        if self.check_visibility(rov_state):
            mult_noise = distance ** (np.log(10) / np.log(2.2))
            self.seen_state = self.true_state[0:3] + camera_noise * mult_noise
            self.tracker.update(self.seen_state)
            self.last_seen_t = 0
        else:
            # If not visible, we update with our own prediction to keep the filter advancing
            measurement = self.tracker.get_state()[0:3]
            self.tracker.update(measurement)
            self.last_seen_t += dt
        
        self.estimated_state = self.tracker.get_state()
        return self.estimated_state

    def truth_update(self, state_moving):
        self.true_state = np.array(state_moving[0:6])

    def get_time_since_seen(self):
        return self.last_seen_t


class KalmanFilterCV:
    """ Basic Constant Velocity Linear Kalman Filter for IMM node. """
    def __init__(self, dt, process_noise, measure_noise):
        self.n_states = 6
        self.n_meas = 3
        self.dt = dt

        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states)

        self.F = np.eye(self.n_states)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        self.H = np.zeros((self.n_meas, self.n_states))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        self.Q = np.eye(self.n_states) * process_noise
        self.R = np.eye(self.n_meas) * measure_noise
        self.I = np.eye(self.n_states)
        
        # Innovation and covariance for IMM probability calculation
        self.y = np.zeros(self.n_meas)
        self.S = np.eye(self.n_meas)

    def set_dt(self, dt):
        self.dt = dt
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        self.y = z - (self.H @ self.x)
        self.S = self.H @ self.P @ self.H.T + self.R

        try:
            K = self.P @ self.H.T @ np.linalg.inv(self.S)
        except np.linalg.LinAlgError:
            K = np.zeros((self.n_states, self.n_meas))

        self.x = self.x + K @ self.y
        self.P = (self.I - K @ self.H) @ self.P


class IMMTracker:
    def __init__(self, dt_default=0.1):
        """
        Interacting Multiple Model Tracker.
        Model 0: Low process noise (Cruising)
        Model 1: High process noise (Maneuvering/Turning)
        """
        self.n_models = 2
        self.n_states = 6
        
        # Initialize the two models with different Q values
        self.models = [
            KalmanFilterCV(dt_default, process_noise=0.01, measure_noise=0.5), # Cruising
            KalmanFilterCV(dt_default, process_noise=1.50, measure_noise=0.5)  # Maneuvering
        ]

        # Markov Transition Probability Matrix (p_ij)
        # Probability of switching from model i to model j
        self.trans_prob = np.array([
            [0.95, 0.05], # From Cruising to -> [Cruising, Maneuvering]
            [0.10, 0.90]  # From Maneuvering to -> [Cruising, Maneuvering]
        ])

        # Mode probabilities (Start with 50/50 or favor cruising)
        self.mu = np.array([0.8, 0.2])

        # Combined state and covariance
        self.x_est = np.zeros(self.n_states)
        self.P_est = np.eye(self.n_states)

    def predict(self, dt=None):
        if dt is not None:
            for m in self.models:
                m.set_dt(dt)

        # 1. Calculate mixing probabilities
        c_bar = np.dot(self.mu, self.trans_prob)
        mu_mix = np.zeros((self.n_models, self.n_models))
        for i in range(self.n_models):
            for j in range(self.n_models):
                mu_mix[i, j] = (self.trans_prob[i, j] * self.mu[i]) / c_bar[j]

        # 2. Mix states and covariances for each model
        x_mixed = np.zeros((self.n_models, self.n_states))
        P_mixed = np.zeros((self.n_models, self.n_states, self.n_states))

        for j in range(self.n_models):
            for i in range(self.n_models):
                x_mixed[j] += self.models[i].x * mu_mix[i, j]

            for i in range(self.n_models):
                diff = self.models[i].x - x_mixed[j]
                P_mixed[j] += mu_mix[i, j] * (self.models[i].P + np.outer(diff, diff))

        # 3. Apply mixed initial conditions and predict
        for j in range(self.n_models):
            self.models[j].x = x_mixed[j]
            self.models[j].P = P_mixed[j]
            self.models[j].predict()

    def update(self, measurement):
        # 1. Update each model with the new measurement
        for m in self.models:
            m.update(measurement)

        # 2. Calculate model likelihoods based on innovation (residual)
        likelihoods = np.zeros(self.n_models)
        for j in range(self.n_models):
            y = self.models[j].y
            S = self.models[j].S
            try:
                # Multivariate Gaussian PDF evaluating the residual
                det_S = np.linalg.det(S)
                inv_S = np.linalg.inv(S)
                exponent = -0.5 * y.T @ inv_S @ y
                likelihoods[j] = (1.0 / np.sqrt((2 * np.pi)**3 * det_S)) * np.exp(exponent)
            except np.linalg.LinAlgError:
                likelihoods[j] = 1e-6

        # 3. Update mode probabilities
        c_bar = np.dot(self.mu, self.trans_prob)
        self.mu = likelihoods * c_bar
        
        # Normalize probabilities to sum to 1
        mu_sum = np.sum(self.mu)
        if mu_sum > 0:
            self.mu /= mu_sum
        else:
            self.mu = np.array([0.5, 0.5]) # Fallback if perfectly zeroed

        # 4. Combine states and covariances for the final output
        self.x_est = np.zeros(self.n_states)
        for j in range(self.n_models):
            self.x_est += self.models[j].x * self.mu[j]

        self.P_est = np.zeros((self.n_states, self.n_states))
        for j in range(self.n_models):
            diff = self.models[j].x - self.x_est
            self.P_est += self.mu[j] * (self.models[j].P + np.outer(diff, diff))

    def get_state(self):
        return self.x_est

    def set_state(self, new_x_est):
        self.x_est = new_x_est.copy()
        for m in self.models:
            m.x = new_x_est.copy()