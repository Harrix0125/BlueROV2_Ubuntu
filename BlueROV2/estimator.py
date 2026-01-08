import numpy as np
import casadi as cas
from nmpc_params import NMPC_params as MPCC
from model import export_bluerov_model

class EKF():
    def __init__(self):
        self.dt = MPCC.T_s
        self.n_states = 12
        self.n_controls = 8
        self.initialized = False
        # 1. Setup CasADi Functions for EKF
        self._setup_casadi_functions()
        
        # 2. Tuning Matrices (Covariances)
        # Q: Process Noise (Trust in physics model)
        #    High Q = Physics is uncertain, rely more on sensors
        #    Low Q = Physics is perfect, ignore noisy sensors
        q_diag = [0.05]*3 + [0.05]*3 + [0.5]*3 + [0.5]*3 # Pos, Att, Vel, Rates
        self.Q = np.diag(q_diag)

        # R: Measurement Noise (Trust in sensors)
        #    High R = Sensors are noisy, rely on physics
        #    Low R = Sensors are precise
        #    Assuming measurement is full state [x,y,z, phi,theta,psi, u,v,w, p,q,r]
        #    Realistically, GPS noise is ~1.0m, IMU is ~0.01 rad
        r_diag = [0.5]*3 + [0.02]*3 + [0.1]*3 + [0.05]*3 
        self.R = np.diag(r_diag)
        
        # 3. Initial State & Covariance
        self.x_est = np.zeros(self.n_states)
        self.P_est = np.eye(self.n_states)*2

    def _setup_casadi_functions(self):
        acados_model = export_bluerov_model()
        x_sym = acados_model.x
        u_sym = acados_model.u
        f_expl = acados_model.f_expl_expr

        # Discretize (Euler Integration) for Prediction Step
        # x_next = x + f(x,u)*dt
        x_next = x_sym + f_expl * self.dt
        
        # Calculate Jacobian F = d(x_next)/dx
        # This is the "A" matrix in linear Kalman Filters
        jac_F_sym = cas.jacobian(x_next, x_sym)

        # Create a fast CasADi function to call in the loop
        # Input: [x, u] -> Output: [x_next, F_matrix]
        self.predict_dynamics = cas.Function('ekf_pred',[x_sym, u_sym],[x_next, jac_F_sym])

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
            self.x_est = measurement
            self.initialized = True
            print("EKF Initialized")
            return self.x_est

        H = np.eye(self.n_states) 
        
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
    
# Will add asyncronous update and predict calls later
# will switcht to quaternions if needed
# Tackle the H Matrix reality: currently,assuming measurement always has 12 items, but the GPS will likely only give 2 or 3 items, and Depth sensor only 1).