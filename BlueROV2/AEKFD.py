import numpy as np
import casadi as cas
from nmpc_params import NMPC_params as MPCC
from model import export_bluerov_model
import utils

class AEKFD():
    def __init__(self):
        self.dt = MPCC.T_s
        self.n_og_states = 12
        self.n_controls = 8
        self.n_dist = 6
        self.n_states = self.n_og_states + self.n_dist

        self.initialized = False

        # Tuning Matrices (Covariances)
        # Q: Process Noise (Trust in physics model)
        #    High Q = Physics is uncertain, rely more on sensors
        #    Low Q = Physics is perfect, ignore noisy sensors
        q_pos = [0.05]*3
        q_att = [0.05]*3
        q_vel = [0.5]*3
        q_rates = [0.5]*3

        q_dist = [0.1]*6  # Disturbance states
        q_diag = q_pos + q_att + q_vel + q_rates + q_dist
        self.Q = np.diag(q_diag)

        # R: Measurement Noise (Trust in sensors)
        #    High R = Sensors are noisy, rely on physics
        #    Low R = Sensors are precise
        #    Assuming measurement is full state [x,y,z, phi,theta,psi, u,v,w, p,q,r]
        #    Realistically, GPS noise is ~1.0m, IMU is ~0.01 rad
        r_pos = [0.5]*3
        r_att = [0.02]*3
        r_vel = [0.1]*3
        r_rates = [0.05]*3
        r_diag = r_pos + r_att + r_vel + r_rates
        self.R = np.diag(r_diag)
        
        # Initial State & Covariance
        self.x_est = np.zeros(self.n_states)
        self.P_est = np.eye(self.n_states)*1

        self._setup_augmented_model()

    def _setup_augmented_model(self):
        """
        Setup AEKF dynamics w/ Acados model with disturbance states, mapping:
        AEKF State [0:12] : Model state x
        AEKF State [12:18] : Disturbance states d
        """
        acados_model = export_bluerov_model()

        model_x = acados_model.x
        model_u = acados_model.u
        model_p = acados_model.p  # Disturbance parameter
        model_rhs = acados_model.f_expl_expr

        # AEKF:
        x_aug = cas.SX.sym('x_aug', self.n_states)
        u_in = cas.SX.sym('u_in', self.n_controls)

        x_phys = x_aug[0:self.n_og_states]
        x_dist = x_aug[self.n_og_states:self.n_states]

        # Substitute dist into the model eqnuations
        x_dot_phys = cas.substitute(model_rhs, cas.vertcat(model_x, model_u, model_p), cas.vertcat(x_phys, u_in, x_dist))

        d_dot = cas.SX.zeros(self.n_dist)  # Disturbance states are constant (zero dynamics)

        x_dot_aug = cas.vertcat(x_dot_phys, d_dot)

        x_next = x_aug + x_dot_aug * self.dt

        jac_F_sym = cas.jacobian(x_next, x_aug)

        self.predict_dynamics = cas.Function('aekf_pred',[x_aug, u_in],[x_next, jac_F_sym])


    def predict(self, control):
        # EKF Prediction Step
        x_pred, F_matrix = self.predict_dynamics(self.x_est, control)
        x_pred = np.array(x_pred).flatten()
        F_matrix = np.array(F_matrix)

        # Update Covariance
        P_pred = F_matrix @ self.P_est @ F_matrix.T + self.Q
        self.x_est = x_pred
        self.P_est = P_pred

        return self.x_est
    

    def measurement_update(self, measurement):
        if not self.initialized:
            self.x_est[0:self.n_og_states] = measurement

            self.initialized = True
            print("EKF Initialized")
            return self.x_est

        # [ I_12x12, 0_12x6 ]
        H = np.zeros((self.n_og_states, self.n_states))
        H[:, :self.n_og_states] = np.eye(self.n_og_states)
        
        # Innovation
        y_k = measurement - H @ self.x_est
        
        # IMPORTANT: Handle angle wrapping for Yaw (Index 5)
        y_k[5] = (y_k[5] + np.pi) % (2 * np.pi) - np.pi

        # S = How much uncertainty total? (State P + Sensor R)
        S_k = H @ self.P_est @ H.T + self.R
        
        # We pass y_k (already wrapped) and S_k (includes R)
        if self.check_outlier(y_k, S_k, threshold=26.2): # Threshold for ~12 DOF
            # REJECT: Return the predicted state as-is
            print("Outlier rejected")
            return self.x_est

        # Calculate Kalman Gain
        K_k = self.P_est @ H.T @ np.linalg.inv(S_k)
        
        # Update State
        self.x_est = self.x_est + K_k @ y_k
        
        # Update Covariance # Search Joseph form as it should be more stable
        I = np.eye(self.n_states)
        self.P_est = (I - K_k @ H) @ self.P_est

        return self.x_est
    
    def check_outlier(self, y_k, S_k, threshold):
        # Calculate Mahalanobis Distance squared: d^2 = y^T * S^-1 * y
        # To add in EKF LaTeX notes
        d_squared = y_k.T @ np.linalg.inv(S_k) @ y_k
        
        if d_squared > threshold:
            print("Outlier detected with d^2 =", d_squared)
            return True  # Is Outlier
        return False     # Is Safe

    def get_disturbance_estimate(self):
        return self.x_est[self.n_og_states:self.n_states]
    
    def get_state_estimate(self):
        return self.x_est[0:self.n_og_states]
    
    def set_state_estimate(self, x_new):
        self.x_est[0:self.n_og_states] = x_new

    def set_disturbance_estimate(self, d_new):
        self.x_est[self.n_og_states:self.n_states] = d_new





        