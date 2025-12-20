import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
from nmpc_params import NMPC_params as MPCC
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

    coriolis_sum = C_rb + C_a

    return coriolis_sum

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

    #coriolis_sum = get_C_np(nu)
    #coriolis_force = coriolis_sum @ nu

    # 3. Acceleration (M * acc = Sum_Forces)
    total_force = tau - damping_force - g_force 
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

def get_linear_traj(start, end, steps, steps_done=0):
    """
    Generates a linear trajectory from start to end in given steps.
    """
    traj = np.zeros((len(start), steps-steps_done))
    for i in range(len(start)):
        moving_start = (end[i] - start[i])/10
        traj[i,:] = np.linspace(moving_start, end[i], steps-steps_done)
    return traj

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
    
    # Initial Conditions (Start at 0,0, -5m depth)
    states[0, 2] = -5.0 
    
    # Simulation Variables
    current_x, current_y, current_z = 0.0, 0.0, -5.0
    current_psi = 2.0  # Yaw
    current_theta = 1.0 # Pitch
    
    for k in range(steps - 1):
        # --- 1. Generate Control Inputs (Steering) ---
        # We simulate "commands" to turn the target
        # Randomize Yaw Rate (r) - Turning left/right
        target_r = np.random.normal(-0.1, 0.1) 
        
        # Randomize Pitch Rate (q) - Diving/Surfacing
        # Keep it small and spring-loaded to return to horizon
        target_q = np.random.normal(0.0, 0.1) - (current_theta * 0.1)
        
        # Roll (p) is usually 0 for a stable sub
        target_p = 0.0
        
        # Surge Speed (u) - Mostly constant forward motion
        target_u = speed + np.random.normal(0.0, 0.05)
        
        # Sway (v) and Heave (w) are 0 (assuming sub moves forward)
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

