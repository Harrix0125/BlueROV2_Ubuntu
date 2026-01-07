import numpy as np

class TargetTrackerKF:
    def __init__(self, dt_default=0.1, process_noise=0.1, measure_noise=0.1):
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
        self.R = np.eye(self.n_meas) * measure_noise

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