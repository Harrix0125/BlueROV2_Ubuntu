import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
from nmpc_params import NMPC_params as MPCC
import utils as utils

def get_error_dynamics(model, target_state):
    """
    model: The AcadosModel object you created (to access model.x)
    target_state: Symbolic variable (SX) for the target's state [x, y, z, phi, theta, psi, u, v, w, p, q, r]
    """
    
    # 1. Unpack ROV State (from model.x)
    # Positions (not needed for rotation, but needed for error calc)
    p_rov = model.x[0:3]
    # Attitudes (phi, theta, psi)
    eta_rov = model.x[3:6]
    # Velocities (u, v, w)
    nu_rov = model.x[6:9] 
    
    # 2. Unpack Target State (passed as an argument or parameter)
    p_target = target_state[0:3]
    eta_target = target_state[3:6] # [phi_t, theta_t, psi_t]
    nu_target = target_state[6:9]  # [u_t, v_t, w_t]
    omega_target = target_state[9:12] # [p_t, q_t, r_t] Angular velocities
    
    # 3. Get Rotation Matrices using YOUR utils function
    # R_rov: Rotates ROV Body -> World
    R_rov = utils.get_J1(eta_rov[0], eta_rov[1], eta_rov[2])
    
    # R_t: Rotates Target Body -> World
    R_t = utils.get_J1(eta_target[0], eta_target[1], eta_target[2])
    
    # 4. Calculate Position Error in Target Frame
    # epsilon = R_t.T * (p_rov - p_target)
    p_error_global = p_rov - p_target
    epsilon = cas.mtimes(R_t.T, p_error_global)

    # 5. Define Skew-Symmetric Matrix for Target Angular Velocity (S(w))
    # This is needed for the rotational derivative part: -S(w) * epsilon
    S_omega = cas.SX.zeros(3, 3)
    S_omega[0, 1] = -omega_target[2] # -r
    S_omega[0, 2] =  omega_target[1] #  q
    S_omega[1, 0] =  omega_target[2] #  r
    S_omega[1, 2] = -omega_target[0] # -p
    S_omega[2, 0] = -omega_target[1] # -q
    S_omega[2, 1] =  omega_target[0] #  p

    # 6. Calculate Error Dynamics (epsilon_dot)
    # eq: dot_epsilon = -S(w)*epsilon + (R_t.T * R_rov * nu_rov) - nu_target
    
    # Term 1: Rotation effect due to target turning
    term_1 = -cas.mtimes(S_omega, epsilon)
    
    # Term 2: Velocity difference projected into Target Frame
    # R_rel = R_t^T * R_rov
    R_rel = cas.mtimes(R_t.T, R_rov) 
    
    # Velocity of ROV seen in Target Frame
    v_rov_in_target_frame = cas.mtimes(R_rel, nu_rov)
    
    term_2 = v_rov_in_target_frame - nu_target
    
    epsilon_dot = term_1 + term_2
    
    return epsilon, epsilon_dot

def get_standoff_reference(rov_state, target_state, desired_dist=2.0, lookahead=1.0, time_predict=1.8):
    """
    Calculates a 'Virtual Reference' for the NMPC to track.
    This creates a carrot-stick approach to maintain distance.
    """
    # Unpack positions
    x_r, y_r, z_r = rov_state[0], rov_state[1], rov_state[2]
    x_t, y_t, z_t = target_state[0], target_state[1], target_state[2]
    
    # Distance and Bearing to Target
    dx = x_t - x_r
    dy = y_t - y_r
    dz = z_t - z_r
    dist_3d = np.linalg.norm([dx, dy, dz])
    dist_plane = np.linalg.norm([dx, dy])
    pitch_d = np.arctan2(-dz, dist_plane)
    yaw_d = np.arctan2(dy, dx)
    if abs(dist_plane) < 0.01:
        yaw_d = 0.0  # Prevent NaN when directly above/below
    #elif abs(dist_3d) < 2:
    #    yaw_d = yaw_d**3  # Slow down yaw changes when very close

    if abs(dist_3d) < 0.01:
        pitch_d = 0.0  # Prevent NaN when at same position
    #elif abs(dist_3d) < 2:
    #    pitch_d = pitch_d**3  # Slow down pitch changes when very close

    # Distance Controller?
    u_speed = 1.3
    w_speed = 1.3
    z_speed = w_speed * (dz)
    x_speed = u_speed * (dist_3d - desired_dist) 

    u_des = np.cos(rov_state[4])*x_speed + np.sin(rov_state[4])*z_speed
    w_des = -np.sin(rov_state[4])*x_speed + np.cos(rov_state[4])*z_speed

    # Clamp speed for safety (e.g., max 0.5 m/s)
    #u_des = np.clip(u_des, -0.9, 1.9)
    
    # Line of Sight: simply look at the target
    psi_des = yaw_d
    
    # Create the "Virtual Carrot" Position
    # We place a reference point 'lookahead' meters away in the desired direction
    # This tricks the Position-Weighted NMPC into moving that way.
    x_ref = x_r + lookahead * np.cos(psi_des)
    y_ref = y_r + lookahead * np.sin(psi_des)
    
    # If we want to STOP (u_des ~ 0), we should set reference to current position
    # But if we want to move, we project it. 
    # A cleaner way for NMPC: Set ref pos based on u_des
    x_ref = x_r + (x_speed * time_predict) * np.cos(psi_des)
    y_ref = y_r + (x_speed * time_predict) * np.sin(psi_des)
    z_ref = z_r + (z_speed * time_predict)
    # But to maintain depth relative to target, we can also do:

    # Build the Reference State Vector (12,)
    # [x, y, z, phi, theta, psi, u, v, w, p, q, r]
    ref_state = np.zeros(12)
    ref_state[0] = x_ref
    ref_state[1] = y_ref
    ref_state[2] = target_state[2] # Match target depth
    ref_state[2] = z_ref
    ref_state[3] = 0 # Roll 0
    ref_state[4] = pitch_d
    ref_state[5] = psi_des # DESIRED HEADING
    ref_state[6] = u_des   # DESIRED SURGE

    ref_state[8] = w_des  # Desired Heave


    ref_state[11] = (psi_des-rov_state[5]) / 10

    # Rest are zero
    return ref_state

def get_J1(phi, theta, psi):
    """Calculates Rotation Matrix (Body -> World) for kinematics"""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    r11 = cpsi*cth
    r12 = -spsi*cphi + sphi*sth*cpsi
    r13 = spsi*sphi + sth*cpsi*cphi
    
    r21 = spsi*cth
    r22 = cpsi*cphi + sphi*sth*spsi
    r23 = -cpsi*sphi + sth*spsi*cphi
    
    r31 = -sth
    r32 = sphi*cth
    r33 = cphi*cth
    row1 = cas.horzcat(r11,r12,r13)
    row2 = cas.horzcat(r21,r22,r23)
    row3 = cas.horzcat(r31,r32,r33)

    J1 = cas.vertcat(row1,row2,row3)
    return J1

def get_J2(phi, theta, psi):
    ttheta = cas.tan(theta)
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth = np.cos(theta)
    J2 = cas.vertcat(
        cas.horzcat(1, sphi*ttheta, cphi*ttheta),
        cas.horzcat(0, cphi, -sphi),
        cas.horzcat(0, sphi/cth, cphi/cth)
    )
    return J2
import numpy as np

def get_J1_np(phi, theta, psi):
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

def get_J2_np(phi, theta, psi):
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
def get_C_cas(nu):
    C_rb = cas.MX.zeros(6, 6)
    # Row 0
    C_rb[0, 4] =  MPCC.m * nu[2];  C_rb[0, 5] = -MPCC.m * nu[1]
    # Row 1
    C_rb[1, 3] = -MPCC.m * nu[2];  C_rb[1, 5] =  MPCC.m * nu[0]
    # Row 2
    C_rb[2, 3] =  MPCC.m * nu[1];  C_rb[2, 4] = -MPCC.m * nu[0]
    C_rb[3, 1] =  MPCC.m * nu[2];  C_rb[3, 2] = -MPCC.m * nu[1]; C_rb[3, 4] = MPCC.Iz * nu[5]; C_rb[3, 5] = -MPCC.Iy * nu[4]
    C_rb[4, 0] = -MPCC.m * nu[2];  C_rb[4, 2] =  MPCC.m * nu[0]; C_rb[4, 3] = -MPCC.Iz * nu[5]; C_rb[4, 5] = MPCC.Ix * nu[3]
    C_rb[5, 0] =  MPCC.m * nu[1];  C_rb[5, 1] = -MPCC.m * nu[0]; C_rb[5, 3] = MPCC.Iy * nu[4]; C_rb[5, 4] = -MPCC.Ix * nu[3]

    C_a = cas.MX.zeros(6, 6)
    C_a[0, 4] = -MPCC.Z_wd * nu[2]; C_a[0, 5] =  MPCC.Y_vd * nu[1]
    C_a[1, 3] =  MPCC.Z_wd * nu[2]; C_a[1, 5] = -MPCC.X_ud * nu[0]
    C_a[2, 3] = -MPCC.Y_vd * nu[1]; C_a[2, 4] =  MPCC.X_ud * nu[0]
    C_a[3, 1] = -MPCC.Z_wd * nu[2]; C_a[3, 2] =  MPCC.Y_vd * nu[1]; C_a[3, 4] = -MPCC.N_rd * nu[5]; C_a[3, 5] = MPCC.M_qd * nu[4]
    C_a[4, 0] =  MPCC.Z_wd * nu[2]; C_a[4, 2] = -MPCC.X_ud * nu[0]; C_a[4, 3] =  MPCC.N_rd * nu[5]; C_a[4, 5] = -MPCC.K_pd * nu[3]
    C_a[5, 0] = -MPCC.Y_vd * nu[1]; C_a[5, 1] =  MPCC.X_ud * nu[0]; C_a[5, 3] = -MPCC.M_qd * nu[4]; C_a[5, 4] = MPCC.K_pd * nu[3]

    coriolis_sum = C_rb 

    return coriolis_sum
def get_C_SX(nu):
    """
    Computes Coriolis matrix using strictly CasADi SX types 
    to match the model definition.
    """
    # 1. Create Empty SX Matrix (NOT np.zeros, NOT MX.zeros)
    C_rb = cas.SX.zeros(6, 6)
    
    m = MPCC.m
    # Use explicit float casting for constants to be safe
    Ix, Iy, Iz = float(MPCC.Ix), float(MPCC.Iy), float(MPCC.Iz)
    
    # 2. Fill Rigid Body Coriolis (Standard SNAME)
    # Note: We access nu by index directly
    u, v, w = nu[0], nu[1], nu[2]
    p, q, r = nu[3], nu[4], nu[5]

    # Row 0
    C_rb[0, 4] =  m * w;   C_rb[0, 5] = -m * v
    # Row 1
    C_rb[1, 3] = -m * w;   C_rb[1, 5] =  m * u
    # Row 2
    C_rb[2, 3] =  m * v;   C_rb[2, 4] = -m * u
    # Row 3
    C_rb[3, 1] = -m * w;   C_rb[3, 2] =  m * v;   C_rb[3, 4] =  Iz * r;   C_rb[3, 5] = -Iy * q
    # Row 4
    C_rb[4, 0] =  m * w;   C_rb[4, 2] = -m * u;   C_rb[4, 3] = -Iz * r;   C_rb[4, 5] =  Ix * p
    # Row 5
    C_rb[5, 0] = -m * v;   C_rb[5, 1] =  m * u;   C_rb[5, 3] =  Iy * q;   C_rb[5, 4] = -Ix * p
    
    # 3. Added Mass Coriolis (C_a)
    C_a = cas.SX.zeros(6, 6)
    Xud, Yvd, Zwd = float(MPCC.X_ud), float(MPCC.Y_vd), float(MPCC.Z_wd)
    Kpd, Mqd, Nrd = float(MPCC.K_pd), float(MPCC.M_qd), float(MPCC.N_rd)

    # Approximated diagonal Added Mass Coriolis
    # (Check your specific notation signs, these follow Fossen generally)
    a1 = Xud * u
    a2 = Yvd * v
    a3 = Zwd * w
    b1 = Kpd * p
    b2 = Mqd * q
    b3 = Nrd * r

    C_a[0, 5] = -a2;   C_a[0, 4] =  a3
    C_a[1, 5] =  a1;   C_a[1, 3] = -a3
    C_a[2, 4] = -a1;   C_a[2, 3] =  a2

    C_a[3, 5] = -b2;   C_a[3, 4] =  b3;  C_a[3, 2] = -a2;  C_a[3, 1] =  a3
    C_a[4, 5] = -b1;   C_a[4, 3] = -b3;  C_a[4, 2] =  a1;  C_a[4, 0] = -a3
    C_a[5, 4] =  b1;   C_a[5, 3] =  b2;  C_a[5, 1] = -a1;  C_a[5, 0] =  a2

    # Note: C_np had some sign diffs and transposition relative to standard Fossen.
    # Ensure this matches your specific model notation (SNAME vs Fossen 1994).
    
    return C_rb + C_a

def get_C_np(nu):
    C_rb = np.zeros((6, 6))
    # Row 0
    C_rb[0, 4] =  MPCC.m * nu[2];  C_rb[0, 5] = -MPCC.m * nu[1]
    # Row 1
    C_rb[1, 3] = -MPCC.m * nu[2];  C_rb[1, 5] =  MPCC.m * nu[0]
    # Row 2
    C_rb[2, 3] =  MPCC.m * nu[1];  C_rb[2, 4] = -MPCC.m * nu[0]
    C_rb[3, 1] =  MPCC.m * nu[2];  C_rb[3, 2] = -MPCC.m * nu[1]; C_rb[3, 4] = MPCC.Iz * nu[5]; C_rb[3, 5] = -MPCC.Iy * nu[4]
    C_rb[4, 0] = -MPCC.m * nu[2];  C_rb[4, 2] =  MPCC.m * nu[0]; C_rb[4, 3] = -MPCC.Iz * nu[5]; C_rb[4, 5] = MPCC.Ix * nu[3]
    C_rb[5, 0] =  MPCC.m * nu[1];  C_rb[5, 1] = -MPCC.m * nu[0]; C_rb[5, 3] = MPCC.Iy * nu[4]; C_rb[5, 4] = -MPCC.Ix * nu[3]

    C_a = np.zeros((6, 6))
    C_a[0, 4] = -MPCC.Z_wd * nu[2]; C_a[0, 5] =  MPCC.Y_vd * nu[1]
    C_a[1, 3] =  MPCC.Z_wd * nu[2]; C_a[1, 5] = -MPCC.X_ud * nu[0]
    C_a[2, 3] = -MPCC.Y_vd * nu[1]; C_a[2, 4] =  MPCC.X_ud * nu[0]
    C_a[3, 1] = -MPCC.Z_wd * nu[2]; C_a[3, 2] =  MPCC.Y_vd * nu[1]; C_a[3, 4] = -MPCC.N_rd * nu[5]; C_a[3, 5] = MPCC.M_qd * nu[4]
    C_a[4, 0] =  MPCC.Z_wd * nu[2]; C_a[4, 2] = -MPCC.X_ud * nu[0]; C_a[4, 3] =  MPCC.N_rd * nu[5]; C_a[4, 5] = -MPCC.K_pd * nu[3]
    C_a[5, 0] = -MPCC.Y_vd * nu[1]; C_a[5, 1] =  MPCC.X_ud * nu[0]; C_a[5, 3] = -MPCC.M_qd * nu[4]; C_a[5, 4] = MPCC.K_pd * nu[3]

    coriolis_sum = C_rb + C_a

    return coriolis_sum
def get_x_dot(x_state, u_control):
    """
    Calculates x_dot = f(x, u) using purely Numpy math.
    """
    # 1. Unpack State
    eta = x_state[0:6]   # [x, y, z, phi, theta, psi]
    nu  = x_state[6:12]  # [u, v, w, p, q, r]
    
    phi, theta, psi = eta[3], eta[4], eta[5]

    # 2. Forces
    # Thruster Forces
    tau = MPCC.TAM @ u_control
    
    # Linear Damping
    damping_lin = MPCC.D_LIN @ nu
    
    # Quadratic Damping (Element-wise calculation)
    # Using abs() ensures drag always opposes motion
    dq_diag = np.abs(nu) * MPCC.D_QUAD_COEFFS 
    damping_quad = dq_diag * nu 
    
    # Restoring Forces (Gravity/Buoyancy)
    W, B, zg = MPCC.W, MPCC.B, MPCC.zg
    diff = W - B
    
    g_force = np.array([
        -diff * np.sin(theta),
        diff * np.cos(theta) * np.sin(phi),
        diff * np.cos(theta) * np.cos(phi),
        zg * W * np.cos(theta) * np.sin(phi),
        zg * W * np.sin(theta),
        0.0
    ])

    # Coriolis (Using the Numpy version)
    # Ensure get_C_np returns a numpy array, not CasADi
    C_mat = get_C_np(nu) 
    coriolis_force = C_mat @ nu

    # 3. Acceleration Dynamics: M * acc = Sum_Forces
    # Total Force in Body Frame
    total_force = tau - damping_lin - damping_quad - g_force - coriolis_force
    
    # Body Acceleration
    acc = MPCC.M_INV @ total_force

    # 4. Kinematics: Velocities -> Pos/Angle rates
    J1 = get_J1_np(phi, theta, psi)
    J2 = get_J2_np(phi, theta, psi)
    
    pos_dot = J1 @ nu[0:3]
    att_dot = J2 @ nu[3:6]
    
    # Stack result [pos_dot, att_dot, acc]
    return np.concatenate((pos_dot, att_dot, acc))

def robot_plant_step_RK4(x_current, u_control, dt):
    """
    Simulates one time step using Runge-Kutta 4 (RK4).
    Much more stable than Euler for Coriolis forces.
    """
    # RK4 Integration
    k1 = get_x_dot(x_current, u_control)
    k2 = get_x_dot(x_current + 0.5 * dt * k1, u_control)
    k3 = get_x_dot(x_current + 0.5 * dt * k2, u_control)
    k4 = get_x_dot(x_current + dt * k3, u_control)
    
    x_next = x_current + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return x_next

def robot_plant_step(x_current, u_control, dt):
    """
    Simulates the robot moving for one time step using NumPy.
    """
    # 1. Unpack State
    eta = x_current[0:6]  # Pos [x,y,z, phi,theta,psi]
    nu  = x_current[6:12] # Vel [u,v,w, p,q,r] (Body Frame)
 
    # 2. Forces
    # Thruster Forces (Tau = TAM * u)
    tau = MPCC.TAM @ u_control
    
    # Linear Damping (Drag) = D_lin * nu
    damping_force = MPCC.D_LIN @ nu
    
    #   ALSO MISSING C
    # Restoring Forces (Gravity/Buoyancy) - Simplified for test
    W = MPCC.W
    B = MPCC.B
    zg = MPCC.zg
    diff = W - B
    
    g_force = np.array([
        -diff * cas.sin(x_current[5]),
        diff * cas.cos(x_current[5]) * cas.sin(x_current[4]),
        diff * cas.cos(x_current[5]) * cas.cos(x_current[4]),
        zg * W * cas.cos(x_current[5]) * cas.sin(x_current[4]),
        zg * W * cas.sin(x_current[5]),
        0
    ])

    coriolis_sum = get_C_np(nu)
    coriolis_force = coriolis_sum @ nu

    # 3. Acceleration (M * acc = Sum_Forces)
    total_force = tau - damping_force - g_force - coriolis_force
    acc = MPCC.M_INV @ total_force
    
    # 4. Kinematics (Vel Body -> Vel World)
    J1 = get_J1(eta[3], eta[4], eta[5])
    J2 = get_J2(eta[3], eta[4], eta[5])
    pos_dot = J1 @ nu[0:3]
    att_dot = J2 @nu[3:6] 

    # 5. Integration (Euler)
    #x_next = cas.SX.zeros(12)
    x_next = np.zeros(12)
    x_next[0:3] = np.squeeze(eta[0:3] + pos_dot[0:3] * dt)
    x_next[3:6] = np.squeeze(eta[3:6] + att_dot[0:3] * dt)
    x_next[6:12] = np.squeeze(nu + acc * dt)
    
    return x_next

def get_linear_traj(steps, dt, speed):
    """
    Generates a linear trajectory from start to end in given steps.
    """
    states = np.zeros((steps, 12))
    
    states[0,0] = 1.8
    states[0,1] = -3
    states[0,2] = -3.0

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

def generate_target_trajectory(steps, dt, speed):
    """
    Generates a 12-state trajectory for MPC testing.
    Returns: numpy array of shape (steps, 12)
    States: [x, y, z, phi, theta, psi, u, v, w, p, q, r]
    """
    
    # Initialize State Matrix (steps x 12)
    # Col 0-2: Pos (x, y, z)
    # Col 3-5: Angle (phi, theta, psi)
    # Col 6-8: LinVel Body (u, v, w)
    # Col 9-11: AngVel Body (p, q, r)
    states = np.zeros((steps, 12))
    
    # Initial Conditions
    states[0,0] = 1.8
    states[0,1] = -3
    states[0,2] = -3.0

    states[0,4] = 0.5  # Pitch
    states[0,5] = 0.0  # Yaw
    # Simulation Variables
    current_x, current_y, current_z = states[0, 0], states[0, 1], states[0, 2]
    current_psi = states[0, 5]  # Yaw
    current_theta = states[0, 4] # Pitch

    for k in range(steps - 1):
        # --- 1. Generate Control Inputs (Steering) ---
        # We simulate "commands" to turn the target
        # Randomize Yaw Rate (r) - Turning left/right
        target_r = -0.1*np.sin(0.004*k) + 0.08*np.sin(0.0003*k)  #turning L/R

        # Randomize Pitch Rate (q) - Diving/Surfacing
        # Keep it small and spring-loaded to return to horizon
        target_q = np.random.normal(0.0, 0.1) - (current_theta * 0.1)
        
        # Roll (p) is usually 0 for a stable target
        target_p = 0.0
        
        # Surge Speed (u) - Mostly constant forward motion
        target_u = speed + np.random.normal(0.0, 0.05)
        
        # Sway (v) and Heave (w) are 0 (assuming target moves forward)
        target_v = 0.0
        target_w = 0.0

        # --- 2. Fill Velocity States (Body Frame) ---
        # In a real dynamic model, forces cause these. 
        # Here we just set them kinematically for the reference.
        states[k+1, 6] = target_u
        states[k+1, 7] = target_v
        states[k+1, 8] = target_w
        states[k+1, 9] = target_p
        states[k+1, 10] = target_q
        states[k+1, 11] = target_r

        # --- 3. Update Angles (Euler Integration) ---
        # Update Yaw and Pitch based on rates
        current_psi += target_r * dt
        current_theta += target_q * dt
        
        states[k+1, 3] = 0.0 # Roll (phi)
        states[k+1, 4] = current_theta
        states[k+1, 5] = current_psi

        # --- 4. Update Position (World Frame) ---
        # We must rotate Body Velocity (u,v,w) into World Velocity (dx, dy, dz)
        # Using standard Rotation Matrix for Yaw (psi) and Pitch (theta)
        
        # Simplified Rotation (assuming small Roll)
        # dx = u * cos(theta) * cos(psi)
        dx = target_u * np.cos(current_theta) * np.cos(current_psi)
        
        # dy = u * cos(theta) * sin(psi)
        dy = target_u * np.cos(current_theta) * np.sin(current_psi)
        
        # dz = -u * sin(theta) (Negative because Z is Down)
        dz = -target_u * np.sin(current_theta)

        current_x += dx * dt
        current_y += dy * dt
        current_z += dz * dt
        
        states[k+1, 0] = current_x
        states[k+1, 1] = current_y
        states[k+1, 2] = current_z

    return states

def get_error_avg_std(state_estimate, target_state, ref_state):
    """
    Calculates Mean Absolute Error (MAE) and Euclidean distance.
    Arguments must be convertible to numpy arrays of shape (3, N).
    """
    # 1. Convert everything to consistent NumPy arrays (3, N)
    #    Rows = x, y, z; Cols = Time steps
    est = np.array(state_estimate) 
    tgt = np.array(target_state)
    ref = np.array(ref_state)

    # Safety check for shapes
    if est.shape != tgt.shape:
        # Handle case where target might be transposed compared to estimate
        if est.shape == tgt.T.shape:
            tgt = tgt.T
        else:
            print(f"Shape mismatch! Est: {est.shape}, Tgt: {tgt.shape}")
            return

    # 2. Calculate Errors for TARGET
    # Difference at every step
    diff_matrix = est - tgt
    
    # A. Per-axis Mean Absolute Error (MAE)
    # Average across time (axis 1)
    mae_x, mae_y, mae_z = np.mean(np.abs(diff_matrix), axis=1)
    
    # B. 3D Euclidean Distance (Average tracking error)
    # Norm at each step, then mean
    dist_3d = np.linalg.norm(diff_matrix, axis=0)
    avg_3d_dist = np.mean(dist_3d)

    # 3. Calculate Errors for REFERENCE (Virtual Carrot)
    diff_ref = est - ref
    dist_ref_3d = np.linalg.norm(diff_ref, axis=0)
    avg_ref_dist = np.mean(dist_ref_3d)

    # 4. Print Results
    print("-" * 30)
    print(f"TRACKING PERFORMANCE:")
    print(f"  Avg 3D Error: {avg_3d_dist:.4f} m")
    print(f"  MAE X: {mae_x:.4f} m")
    print(f"  MAE Y: {mae_y:.4f} m")
    print(f"  MAE Z: {mae_z:.4f} m")
    print(f"  Avg Dist from Ref: {avg_ref_dist:.4f} m")
    print("-" * 30)

    # Return dictionary for logging if needed
    return {
        'mae_x': mae_x,
        'mae_y': mae_y,
        'mae_z': mae_z,
        'avg_3d': avg_3d_dist
    }
