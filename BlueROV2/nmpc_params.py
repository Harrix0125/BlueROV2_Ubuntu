import numpy as np
import casadi as cas

class NMPC_params:
    N = 280             # Prediction horizon (steps)
    T_s = 0.03      # Time step (seconds) - 20Hz is standard for underwater
    
    m = 11.5            # mass [kg]
    W = 112.8           # weight [N]
    B = 114.8           # buoyancy [N]
    g = 9.82            # gravity [m/s^2]
    
    # Center of Gravity (CG) and Buoyancy (CB)
    # CG relative to CO
    r_g = np.array([0.0, 0.0, 0.02]) 
    xg, yg, zg = r_g[0], r_g[1], r_g[2]
    
    # CB relative to CO
    r_b = np.array([0.0, 0.0, -0.1])
    xb, yb, zb = r_b[0], r_b[1], r_b[2]

    # Moments of Inertia
    Ix = 0.16
    Iy = 0.16
    Iz = 0.16

    # Added Mass (Diagonal terms)
    X_ud = -5.5
    Y_vd = -12.7
    Z_wd = -14.57
    K_pd = -0.12
    M_qd = -0.12
    N_rd = -0.12

    # Linear Damping
    X_u = -4.03
    Y_v = -6.22
    Z_w = -5.18
    K_p = -0.07
    M_q = -0.07
    N_r = -0.07

    # Quadratic Damping (Drag)
    X_uu = -18.18
    Y_vv = -21.66
    Z_ww = -36.99
    K_pp = -1.55
    M_qq = -1.55
    N_rr = -1.55
    
    # Rigid Body Mass Matrix (Mrb)
    # MATLAB: [m, 0, 0, 0, m*zg, 0; ...]
    Mrb = np.array([
        [m, 0, 0, 0, m*zg, -m*yg],
        [0, m, 0, -m*zg, 0, m*xg],
        [0, 0, m, m*yg, -m*xg, 0],
        [0, -m*zg, m*yg, Ix, 0, 0],
        [m*zg, 0, -m*xg, 0, Iy, 0],
        [-m*yg, m*xg, 0, 0, 0, Iz]
    ])

    # Added Mass Matrix (Ma)
    Ma = -np.diag([X_ud, Y_vd, Z_wd, K_pd, M_qd, N_rd])

    # Total Mass Matrix (M)
    M = Mrb + Ma
    
    # Numerical stability (Epsilon)
    epsilon = 1e-4
    M = M + epsilon * np.eye(6)
    
    # Inverse mass matrix
    M_INV = np.linalg.inv(M)

    # Linear Damping Matrix (D_lin)
    D_LIN = -np.diag([X_u, Y_v, Z_w, K_p, M_q, N_r])
    
    # Quadratic Damping Matrix (D_quad)
    D_QUAD_COEFFS = np.array([-X_uu, -Y_vv, -Z_ww, -K_pp, -M_qq, -N_rr])

    # Thruster Allocation Matrix - 6x8
    TAM = np.array([
        [ 0.707,  0.707, -0.707, -0.707,  0.0,    0.0,    0.0,    0.0   ],
        [-0.707,  0.707, -0.707,  0.707,  0.0,    0.0,    0.0,    0.0   ],
        [ 0.0,    0.0,    0.0,    0.0,   -1.0,    1.0,    1.0,   -1.0   ],
        [ 0.06,  -0.06,   0.06,  -0.06,  -0.218, -0.218,  0.218,  0.218 ],
        [ 0.06,   0.06,  -0.06,  -0.06,   0.120, -0.120,  0.120, -0.120 ],
        [-0.1888, 0.1888, 0.1888, -0.1888, 0.0,   0.0,    0.0,    0.0   ]
    ])

    THRUST_MIN = -30.0
    THRUST_MAX = 30.0
    
    # Tuning Weights
    # Position Errors [x, y, z, phi, theta, psi]
    pos_coef = 6.0
    angle_coef = 0.0
    Q_POS = [pos_coef, pos_coef, pos_coef, angle_coef, angle_coef, angle_coef] 
    
    # Velocity Errors [u, v, w, p, q, r]
    vel_coef = 2
    angV_coef = 2
    Q_VEL = [vel_coef, vel_coef, vel_coef, angV_coef, angV_coef, angV_coef]
    
    # Control Effort (Minimize energy)
    R_THRUST = 0.02
    Q_diag = Q_POS + Q_VEL
    Q = cas.diag(Q_diag)

    #   Weights at time = N
    pos_N = 500.0
    angle_N = 1.0
    Q_POS_N = [pos_N, pos_N, pos_N, angle_N, angle_N, angle_N] 

    vel_N = 1
    angV_N = 1
    Q_VEL_N = [vel_N, vel_N, vel_N, angV_N, angV_N, angV_N]
    Q_diag_N = Q_POS_N + Q_VEL_N
    Q_N = cas.diag(Q_diag_N)