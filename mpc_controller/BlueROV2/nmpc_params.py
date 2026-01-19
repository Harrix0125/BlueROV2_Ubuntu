import numpy as np
import casadi as cas

class NMPC_params:

    N = 180          # Prediction horizon (steps)
    T_s = 0.05      # Time step (seconds) - between 10-20Hz ideally

    def __init__(self):
        self.nx = 12
        self.nu = 8

class BlueROV_Params(NMPC_params):
    def __init__(self):
        super().__init__()
        self.m = 11.5            # self.mass [kg]
        self.W = 112.8           # weight [N]
        self.B = 114.8           # buoyancy [N]
        self.g = 9.82            # gravity [m/s^2]
        # For gazebo data (It would be easy to get data for real world ROV aswell, at least for the mass as it is sufficient to weight it and there would be low error)
        self.m = 13
        self.W = self.m * self.g
        self.B = 13.14 * self.g

        self.fov_h = 90
        self.fov_v = 80
        
        # Center of Gravity (CG) and Buoyancy (CB)
        # CG relative to CO
        self.r_g = np.array([0.0, 0.0, -0.02]) 
        self.xg, self.yg, self.zg = self.r_g[0], self.r_g[1], self.r_g[2]
        
        # CB relative to CO
        self.r_b = np.array([0.0, 0.0, 0.1])
        self.xb, self.yb, self.zb = self.r_b[0], self.r_b[1], self.r_b[2]

        # self.moments of Inertia
        self.Ix = 0.16
        self.Iy = 0.16
        self.Iz = 0.16

        # Added self.mass (Diagonal terms)
        self.X_ud = -5.5
        self.Y_vd = -12.7
        self.Z_wd = -14.57
        self.K_pd = -0.12
        self.M_qd = -0.12
        self.N_rd = -0.12
        # Set to zero for Gazebo Simulation! be careful
        self.X_ud = 0
        self.Y_vd = 0
        self.Z_wd = 0
        self.K_pd = 0
        self.M_qd = 0
        self.N_rd = 0

        # Linear Damping
        self.X_u = -4.03
        self.Y_v = -6.22
        self.Z_w = -5.18
        self.K_p = -0.07
        self.M_q = -0.07
        self.N_r = -0.07

        # Quadratic Damping (Drag)
        self.X_uu = -18.18
        self.Y_vv = -21.66
        self.Z_ww = -36.99
        self.K_pp = -1.55
        self.M_qq = -1.55
        self.N_rr = -1.55
        
        # Rigid Body self.mass self.matrix (Mrb)
        # self.mATLAB: [m, 0, 0, 0, self.m*zg, 0; ...]
        self.Mrb = np.array([
            [self.m, 0, 0, 0, self.m*self.zg, -self.m*self.yg],
            [0, self.m, 0, -self.m*self.zg, 0, self.m*self.xg],
            [0, 0, self.m, self.m*self.yg, -self.m*self.xg, 0],
            [0, -self.m*self.zg, self.m*self.yg, self.Ix, 0, 0],
            [self.m*self.zg, 0, -self.m*self.xg, 0, self.Iy, 0],
            [-self.m*self.yg, self.m*self.xg, 0, 0, 0, self.Iz]
        ])

        # Added self.mass self.matrix (Ma)
        self.Ma = -np.diag([self.X_ud, self.Y_vd, self.Z_wd, self.K_pd, self.M_qd, self.N_rd])

        # Total self.mass self.matrix (M)
        self.M = self.Mrb + self.Ma
        # Numerical stability (self.epsilon)
        self.epsilon = 1e-4
        self.M = self.M + self.epsilon * np.eye(6)
        
        # Inverse self.mass self.matrix
        self.M_INV = np.linalg.inv(self.M)

        # Linear Damping self.matrix (self.D_LIN)
        self.D_LIN = -np.diag([self.X_u, self.Y_v, self.Z_w, self.K_p, self.M_q, self.N_r])
        
        # Quadratic Damping self.matrix (D_quad)
        self.D_QUAD_COEFFS = np.array([-self.X_uu, -self.Y_vv, -self.Z_ww, -self.K_pp, - self.M_qq, -self.N_rr])

        # Thruster Allocation self.matrix - 6x8
        # self.TAM = np.array([
        #     [ 0.707,  0.707, -0.707, -0.707,  0.0,    0.0,    0.0,    0.0   ],
        #     [-0.707,  0.707, -0.707,  0.707,  0.0,    0.0,    0.0,    0.0   ],
        #     [ 0.0,    0.0,    0.0,    0.0,   -1.0,    1.0,    1.0,   -1.0   ],
        #     [ 0.06,  -0.06,   0.06,  -0.06,  -0.218, -0.218,  0.218,  0.218 ],
        #     [ 0.06,   0.06,  -0.06,  -0.06,   0.120, -0.120,  0.120, -0.120 ],
        #     [-0.1888, 0.1888, 0.1888, -0.1888, 0.0,   0.0,    0.0,    0.0   ]
        # ])

        self.TAM = np.array([
            [-1*0.707, -1*0.707,  +1*0.707,  +1*0.707,  0.0,    0.0,    0.0,    0.0   ],
            [ -1*0.707, +1*0.707, -1*0.707,  +1*0.707,  0.0,    0.0,    0.0,    0.0   ],
            [ 0.0,    0.0,    0.0,    0.0,              -1.0,   -1.0,   -1.0,    -1.0   ],
            [ -1*0.00,   +1*0.00,  -1*0.00,  +1*0.00,  0.218, -0.218,  0.218,  -0.218 ],
            [ +1*0.00,  +1*0.00,   -1*0.00,  -1*0.00,   0.120, 0.120, -0.120,  -0.120 ],
            [ -1*0.1888,-1*0.1888, +1*0.1888, +1*0.1888, 0.0,   0.0,    0.0,    0.0   ]
        ])

        self.THRUST_MIN = -30.0
        self.THRUST_MAX = 30.0
        self.DELTA_THRUST_LIMIT = self.THRUST_MAX * self.T_s * 2
        
        # Tuning Weights
        # Position Errors [x, y, z, phi, theta, psi]
        # self.pos_coef = 5 #chill
        self.pos_coef = 15
        self.z_coef = self.pos_coef * 3
        self.angle_coef = 1
        self.pitch_coef = 40
        self.psi_coef = 30
        self.Q_POS = [self.pos_coef, self.pos_coef, self.z_coef, self.angle_coef, self.pitch_coef, self.psi_coef] 
        
        # Velocity Errors [u, v, w, p, q, r]
        self.vel_coef =10
        self.angV_coef = 1
        self.Q_VEL = [self.vel_coef, self.vel_coef, self.vel_coef, self.angV_coef, self.angV_coef, self.angV_coef]
        
        # Control Effort to self.minimize thruster usage: if too low it goes crazy and rotates
        self.R_THRUST = 0.01
        self.Q_diag = self.Q_POS + self.Q_VEL
        self.Q = cas.diag(self.Q_diag)

        #   Weights at time = N
        # pos_n = 150  #chill
        self.pos_N = 180.0
        self.angle_N = 80.0
        self.Q_POS_N = [self.pos_N, self.pos_N, self.pos_N*1.5, self.angle_N, self.angle_N, self.angle_N] 

        # vel_N = 10  #chill
        self.vel_N = 15
        self.angV_N = 1.0
        self.Q_VEL_N = [self.vel_N, self.vel_N, self.vel_N, self.angV_N, self.angV_N, self.angV_N]
        self.Q_diag_N = self.Q_POS_N + self.Q_VEL_N
        self.Q_N = cas.diag(self.Q_diag_N)

        # For constraints:
        self.z_min = -100
        self.z_max = 0.5

        # --- TUNING KALMAN FILTER (AEKFD) ---       
        # Tuning Matrices (Covariances)
        # Q: Process Noise (Trust in physics model)
        #    High Q = Physics is uncertain, rely more on sensors
        #    Low Q = Physics is perfect, ignore noisy sensors
        q_pos = [0.5, 0.5, 0.01]
        q_att = [0.8, 0.5, 0.5]
        q_vel = [1.0, 1.0, 0.1]
        q_rates = [0.05, 0.05, 0.1]

        q_dist = [0.01,0.01,0.01,0.001,0.001,0.001]  # Disturbance states
        #q_dist = [0]*6
        q_diag = q_pos + q_att + q_vel + q_rates + q_dist
        self.AEKFD_Q = np.diag(q_diag)

        # R: Measurement Noise (Trust in sensors)
        #    High R = Sensors are noisy, rely on physics
        #    Low R = Sensors are precise
        #    Assuming measurement is full state [x,y,z, phi,theta,psi, u,v,w, p,q,r]
        #    Realistically, GPS noise is ~1.0m, IMU is ~0.01 rad
        r_pos = [0.1]*3
        r_att = [0.01]*3
        r_vel = [0.01, 0.01, 0.01]
        r_rates = [0.01]*3
        r_diag = r_pos + r_att + r_vel + r_rates
        self.AEKFD_R = np.diag(r_diag)

        self.noise_ekf = np.random.normal(0, [0.10, 0.10, 0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

# -------------------------------------------
# --#####------#####---#------#--------------
# --#----##---#-----#--#------#--------------
# --#-----##--#-----#--#------#--------------
# --#----##---#-----#--#------#--------------
# --#####-----#-----#--#------#--------------
# --#----##---#-----#--#------#--------------
# --#-----##--#-----#---#----#---------------
# --#-----##--#-----#----#--#----------------
# --#-----##---#####-------#-----------------

# i had fun making these so i can scroll this i see this :) i dont think that's good coding habit but idc d(^_^)b

# -------------------------------------------
# --#####------#####----#####---#######------
# --#----##---#-----#--#-----#-----#---------
# --#-----##--#-----#--#-----#-----#---------
# --#----##---#-----#--#-----#-----#---------
# --#####-----#-----#--#######-----#---------
# --#----##---#-----#--#-----#-----#---------
# --#-----##--#-----#--#-----#-----#---------
# --#----##---#-----#--#-----#-----#---------
# --#####------#####---#-----#-----#---------

class BlueBoat_Params(NMPC_params):
    def __init__(self):
        super().__init__()
        self.nu = 2  # just 2 thrusters
        
        # random physics for boat
        self.m = 19.5
        self.W = self.m * 9.81
        self.B = self.W  # floats
        self.zg = 0.05   # erm
        
        self.fov_h = 180
        self.fov_v = 180
        # random
        self.Ix = 5.0   # Roll (difficile da ruotare)
        self.Iy = 8.0   # Pitch
        self.Iz = 4.5   # Yaw

        # --- TAM (Thruster Allocation Matrix) ---
        d = 0.435 
        self.TAM = np.array([
            [ 1.0,  1.0],  # Both go ahead
            [ 0.0,  0.0],  # No lateral movement
            [ 0.0,  0.0],  # No vertical movement
            [ 0.0,  0.0],  # Roll
            [ 0.0,  0.0],  # Pitch
            [-d,    d  ]   # Yaw (N): One clockwise, other counterclockwise
        ])
        
        # Needed for Coriolis, they are improvised...
        self.X_ud = 2.0 
        self.Y_vd = 10.0 
        self.Z_wd = 20.0 
        self.K_pd = 1.0
        self.M_qd = 5.0
        self.N_rd = 3.0

        # Linear Damping
        self.D_LIN = -np.diag([3.0, 80.0, 50.0, 20.0, 20.0, 5.0]) 
        # Nota: Y_v (80) to simulate lateral resistance
        
        # Quadratic Damping (imaginary)
        self.D_QUAD_COEFFS = np.array([15.0, 100.0, 100.0, 50.0, 50.0, 15.0])

        # Semplification: needed
        M_diag = np.array([self.m, self.m+10, self.m+20, self.Ix, self.Iy, self.Iz])
        self.M_INV = np.diag(1.0/M_diag)

        # Limiti Motori
        self.THRUST_MIN = -40.0 
        self.THRUST_MAX = 50.0  
        self.R_THRUST = 0.1

        # --- TUNING NMPC ---
        # = 0 what we cant control
        # [x, y, z, phi, theta, psi]
        self.Q_POS = [20, 20, 0.1, 0.1, 0.1, 60]
        # [u, v, w, p, q, r]
        self.Q_VEL = [5, 5, 0.1, 0.1, 0.1, 5.0]
        self.Q_diag = self.Q_POS + self.Q_VEL
        self.Q = cas.diag(self.Q_diag)


        # [x, y, z, phi, theta, psi]
        self.Q_POS_N = [200, 200, 0.1, 0.1, 0.1, 100]
        # [u, v, w, p, q, r]
        self.Q_VEL_N = [20, 20, 0.1, 0.1, 0.1, 10]
        self.Q_diag_N = self.Q_POS_N + self.Q_VEL_N
        self.Q_N = cas.diag(self.Q_diag_N)

        self.z_min = -10 
        self.z_max = 10


        # --- TUNING KALMAN FILTER (AEKFD) ---        
        # Tuning Matrices (Covariances)
        # Q: Process Noise (Trust in physics model)
        #    High Q = Physics is uncertain, rely more on sensors
        #    Low Q = Physics is perfect, ignore noisy sensors
        # Z, Roll, Pitch = 0 for process noise
        q_pos   = [0.5,  0.5,  1e-3]  # x, y, z (z bloccato)
        q_att   = [1e-3, 1e-3, 0.5 ]  # phi, theta, psi (roll/pitch bloccati)
        q_vel   = [0.5,  0.5,  1e-3]  # u, v, w (w bloccato)
        q_rates = [1e-3, 1e-3, 0.5 ]  # p, q, r
        # Disturbance states
        q_dist  = [0.5, 0.5, 1e-3, 1e-3, 1e-3, 0.1] 
        self.AEKFD_Q = np.diag(q_pos + q_att + q_vel + q_rates + q_dist)

        # R: Measurement Noise (Trust in sensors)
        #    High R = Sensors are noisy, rely on physics
        #    Low R = Sensors are precise
        #    Assuming measurement is full state [x,y,z, phi,theta,psi, u,v,w, p,q,r]
        #    Realistically, GPS noise is ~1.0m, IMU is ~0.01 rad
        r_pos   = [0.5,  0.5,  1e-3]
        r_att   = [1e-3, 1e-3, 0.1] 
        r_vel   = [0.1,  0.1,  1e-3]
        r_rates = [0.05, 0.05, 1e-3]
        self.AEKFD_R = np.diag(r_pos + r_att + r_vel + r_rates)

        self.noise_ekf = np.random.normal(0, [0.10, 0.10, 0.00, 0.00, 0.00, 0.02, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01])