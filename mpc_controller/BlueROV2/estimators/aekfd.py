import numpy as np
import casadi as cas

class AEKFD():
    def __init__(self, acados_model, params, x0 = None):
        self.dt = params.T_s
        self.n_og_states = 12
        self.n_controls = params.nu
        self.n_dist = 6
        self.n_states = self.n_og_states + self.n_dist

        self.initialized = False

        # Tuning Matrices (Covariances)
        # Q: Process Noise (Trust in physics model)
        #    High Q = Physics is uncertain, rely more on sensors
        #    Low Q = Physics is perfect, ignore noisy sensors
        self.Q = params.AEKFD_Q

        # R: Measurement Noise (Trust in sensors)
        #    High R = Sensors are noisy, rely on physics
        #    Low R = Sensors are precise
        #    Assuming measurement is full state [x,y,z, phi,theta,psi, u,v,w, p,q,r]
        #    Realistically, GPS noise is ~1.0m, IMU is ~0.01 rad        
        self.R = params.AEKFD_R
        
        # Initial State & Covariance
        self.x_est = np.zeros(self.n_states)
        self.P_est = np.eye(self.n_states)


        self.beta_max = 50.0
        self.gamma = 1.0
        self.nis_threshold = 9.0
        self.use_vff = False
        self.Kf = 3.5
        self.lambda_min = 0.7
        self.lambda_max = 1.0
        self.lambda_k = 1.0

        self._setup_augmented_model(acados_model)

    def _setup_augmented_model(self, acados_model):
        """
        Setup AEKF dynamics w/ Acados model with disturbance states, mapping:
        AEKF State [0:12] : Model state x
        AEKF State [12:18] : Disturbance states d
        """
        

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
        x_dot_phys_expr = cas.substitute(model_rhs, cas.vertcat(model_x, model_u, model_p), cas.vertcat(x_phys, u_in, x_dist))

        d_dot_expr = cas.SX.zeros(self.n_dist)  # Disturbance states are constant (zero dynamics)

        x_dot_aug_expr = cas.vertcat(x_dot_phys_expr, d_dot_expr)

        f_dyn = cas.Function('f_dyn', [x_aug, u_in], [x_dot_aug_expr])

        k1 = f_dyn(x_aug, u_in)
        k2 = f_dyn(x_aug + 0.5*self.dt * k1, u_in)
        k3 = f_dyn(x_aug + 0.5*self.dt * k2, u_in)
        k4 = f_dyn(x_aug + self.dt * k3, u_in)


        x_next = x_aug + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)

        jac_F_sym = cas.jacobian(x_next, x_aug)

        self.predict_dynamics = cas.Function('aekf_pred',[x_aug, u_in],[x_next, jac_F_sym])


    def predict(self, control):
        # EKF Prediction Step
        x_pred, F_matrix = self.predict_dynamics(self.x_est, control)
        x_pred = np.array(x_pred).flatten()
        F_matrix = np.array(F_matrix)

        constrained_indeces =  [2,3,4,8,9,10]
        if self.n_controls == 2:
            x_pred[constrained_indeces] = 0.0

        Q_dist = np.eye(self.n_dist) * 0.01
        Q_aug = self.Q.copy()
        Q_aug[self.n_og_states: , self.n_og_states:] += Q_dist*self.dt

        # Update Covariance
        P_pred = F_matrix @ self.P_est @ F_matrix.T + Q_aug
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

        constrained_indeces =  [2,3,4,8,9,10]
        if self.n_controls == 2:
            y_k[constrained_indeces] = 0.0
        
        # IMPORTANT: Handle angle wrapping for Yaw (Index 5)
        y_k[5] = (y_k[5] + np.pi) % (2 * np.pi) - np.pi

        # S = How much uncertainty total? (State P + Sensor R)
        S_k = H @ self.P_est @ H.T + self.R
        
        #  Adaptive Law from here
        nis = self.check_outlier(y_k, S_k)
        #threshold of 3std deviation
        gamma = 3.0 
        d = np.sqrt(nis)

        # if d <= gamma:
        #     weight = 1.0
        #     # self.adapt_Q(y_k,S_k, nis)
        # else:
        #     # so the weight is just <= 1.0
        #     weight = gamma / d 
        #     print(f"Applying Huber weight: {weight:.3f} (NIS: {nis:.2f})")
        #     # just to see we entered it

        
        if self.use_vff:
            innovation_norm = np.linalg.norm(y_k)
            self.lambda_k = np.exp(-self.Kf * innovation_norm)
            if self.lambda_k < 0.05:
                self.lambda_k = 0.05
            K_k = self.P_est @ H.T @ np.linalg.inv(S_k)
            self.x_est = self.x_est + K_k @ y_k
            I = np.eye(self.n_states)
            P_std = (I - K_k @ H) @ self.P_est @ (I - K_k @ H).T + K_k @ self.R @ K_k.T
            P_agg = (1.0/self.lambda_k) * P_std
            self.P_est = (1 - self.lambda_k) * P_agg + self.lambda_k * P_std

        else:
            K_k = self.P_est @ H.T @ np.linalg.inv(S_k)
            self.x_est = self.x_est + K_k @ y_k
            
            # Update Covariance # Search Joseph form as it should be more stable
            I = np.eye(self.n_states)
            # self.P_est = (I - K_k @ H) @ self.P_est
            #   THIS is Joseph form covariance update, to mathematically guarantee that the cov matrix remains sym & positive semi-def
            #       given recalculation of K it is NOT granted with normal Pcov update
            self.P_est = (I - K_k @ H) @ self.P_est @ (I - K_k @ H).T + K_k @ self.R @ K_k.T
        return self.x_est
    
    def check_outlier(self, y_k, S_k):
        # Calculate Mahalanobis Distance squared: d^2 = y^T * S^-1 * y = Normalized Innovation Squared
        # To add in EKF LaTeX notes
        d_squared = y_k.T @ np.linalg.inv(S_k) @ y_k
        return d_squared
    
    def adapt_Q(self, y_k, S_k, nis):
        """
        Adapting Q by a factor 
        """
        expected_nis = 12
        
        if nis > expected_nis * 1.5:   
            # multiply Q
            scale = 1.0 + 0.1 * (nis / expected_nis - 1.0)
            scale = min(scale, 3.0)   # gotta cap it
            self.Q *= scale
        elif nis < expected_nis * 0.5:
            scale = 0.95
            self.Q *= scale

    def get_disturbance_estimate(self):
        return self.x_est[self.n_og_states:self.n_states]
    
    def get_state_estimate(self):
        return self.x_est[0:self.n_og_states]
    
    def set_state_estimate(self, x_new):
        self.x_est[0:self.n_og_states] = x_new

    def set_disturbance_estimate(self, d_new):
        self.x_est[self.n_og_states:self.n_states] = d_new





        