import numpy as np

def get_shadow_ref(rov_state, target_state, target_vel=None, desired_dist=2.0):
    """ Calculates the 'Ideal State' (Shadow) for the NMPC to track. """
    p_rov = rov_state[0:3]
    p_target = target_state[0:3]
    v_target_global = target_vel if target_vel is not None else np.zeros(3)

    error_vector = p_rov - p_target
    dist_3d = np.linalg.norm(error_vector)

    if dist_3d < 0.01:
        direction_vect = np.array([-1.0, 0.0, 0.0])
    else:
        direction_vect = error_vector / dist_3d

    p_reference = p_target + direction_vect * desired_dist
    
    # Calculate desired yaw to face target
    target_pointing_vector = p_target - p_reference
    yaw_des = np.arctan2(target_pointing_vector[1], target_pointing_vector[0])

    # Unwrapping yaw
    diff = yaw_des - rov_state[5]
    if abs(diff) > np.pi:
        if diff > np.pi: yaw_des -= 2 * np.pi
        elif diff < -np.pi: yaw_des += 2 * np.pi

    dist_plane = np.linalg.norm(target_pointing_vector[0:2])
    pitch_des = np.arctan2(-target_pointing_vector[2], dist_plane)

    # Transform global velocity to body frame reference (simplified)
    c_psi, s_psi = np.cos(yaw_des), np.sin(yaw_des)
    u_ref = v_target_global[0] * c_psi + v_target_global[1] * s_psi
    v_ref = -v_target_global[0] * s_psi + v_target_global[1] * c_psi
    w_ref = v_target_global[2]

    ref_state = np.zeros(12)
    ref_state[0:3] = p_reference
    ref_state[4] = pitch_des
    ref_state[5] = yaw_des
    ref_state[6] = u_ref
    ref_state[7] = v_ref
    ref_state[8] = w_ref

    return ref_state

def get_shadow_traj(rov_state, target_state, target_vel, dt, horizon_N, desired_dist=2.0):
    """ Generates a list of N reference states (The Trajectory) for the NMPC. """
    trajectory = []
    
    # Initial reference based on current target position
    current_ref = get_shadow_ref(rov_state, target_state, target_vel, desired_dist)
    
    v_target_global = target_vel if target_vel is not None else np.zeros(3)

    for k in range(int(horizon_N)):
        current_ref[0:3] += v_target_global * dt
        
        target_pos_future = target_state[0:3] + v_target_global * (k * dt)
        pointing_vec = target_pos_future - current_ref[0:3]
        
        current_ref[5] = np.arctan2(pointing_vec[1], pointing_vec[0]) # Yaw
        dist_plane = np.linalg.norm(pointing_vec[0:2])
        current_ref[4] = np.arctan2(-pointing_vec[2], dist_plane)     # Pitch
        
        psi = current_ref[5]
        theta = current_ref[4]
        
        c_p, s_p = np.cos(psi), np.sin(psi)
        current_ref[6] = v_target_global[0] * c_p + v_target_global[1] * s_p # u
        current_ref[7] = -v_target_global[0] * s_p + v_target_global[1] * c_p # v
        current_ref[8] = v_target_global[2]                                  # w (vertical)

        current_ref[9:12] = 0.0 

        trajectory.append(current_ref.copy())

    return np.array(trajectory)

def get_shadow_traj_ROV_optimized(rov_state, target_state, target_vel, dt, horizon_N, desired_dist=2.5):
    trajectory = []
    
    # 1. Get initial reference
    current_ref = get_shadow_ref(rov_state, target_state, target_vel, desired_dist)
    v_target_global = target_vel if target_vel is not None else np.zeros(3)
    
    # 2. Lock the orientation for the short horizon to prevent NMPC thrashing
    locked_yaw = current_ref[5]
    locked_pitch = current_ref[4]
    
    c_p, s_p = np.cos(locked_yaw), np.sin(locked_yaw)
    u_ref = v_target_global[0] * c_p + v_target_global[1] * s_p
    v_ref = -v_target_global[0] * s_p + v_target_global[1] * c_p
    w_ref = v_target_global[2]

    for k in range(int(horizon_N)):
        ref_state = np.zeros(12)
        
        # Linearly project position based on KF velocity
        ref_state[0:3] = current_ref[0:3] + v_target_global * (k * dt)
        
        # Apply locked orientation
        ref_state[4] = locked_pitch
        ref_state[5] = locked_yaw
        
        # Apply constant body-frame velocities
        ref_state[6] = u_ref
        ref_state[7] = v_ref
        ref_state[8] = w_ref

        trajectory.append(ref_state)

    return np.array(trajectory)

def get_shadow_LOS(rov_state, target_state, target_vel=None, desired_dist=2.0):
    """ Generates Boat shadow leader position """
    p_rov = rov_state[0:3]
    p_target = target_state[0:3]
    v_target_global = target_vel if target_vel is not None else np.zeros(3)

    error_vector = p_rov - p_target
    dist_3d = np.linalg.norm(error_vector)

    if dist_3d < 0.01:
        direction_vect = np.array([-1.0, 0.0, 0.0])
    else:
        direction_vect = error_vector / dist_3d
    
    # p_reference is the closest point from the target such that it is as a desire_dist distance, so at radius of desired_dist
    p_reference = p_target + direction_vect * desired_dist
    
    # Calculate desired yaw to face target
    target_pointing_vector = p_target - p_reference
    yaw_des = np.arctan2(target_pointing_vector[1], target_pointing_vector[0])

    # Unwrapping yaw
    diff = yaw_des - rov_state[5]
    if abs(diff) > np.pi:
        if diff > np.pi: yaw_des -= 2 * np.pi
        elif diff < -np.pi: yaw_des += 2 * np.pi
    
    diff = yaw_des - rov_state[5]
    angular_vel = 0
    if np.abs(diff) < 0.1:
        angular_vel = diff
    else:
        angular_vel = np.sqrt(np.abs(diff))*np.sign(diff)

    
    # Transform global velocity to body frame reference (simplified)
    c_psi, s_psi = np.cos(yaw_des), np.sin(yaw_des)
    u_ref = v_target_global[0]
    v_ref = 0
    w_ref = 0

    ref_state = np.zeros(12)
    ref_state[0:3] = p_reference
    ref_state[4] = 0
    ref_state[5] = yaw_des
    ref_state[6] = u_ref
    ref_state[7] = v_ref
    ref_state[8] = w_ref
    ref_state[11] = angular_vel

    return ref_state

def get_shadow_traj_BOAT(boat_state, target_state, target_vel, dt, horizon_N, desired_dist=2.5):
    """ Generates a dynamically feasible N-step trajectory for an underactuated boat. """
    trajectory = []
    
    # Force 2D plane constraints
    p_boat = boat_state[0:3].copy()
    p_boat[2] = 0.0 
    
    p_target = target_state[0:3].copy()
    p_target[2] = 0.0
    
    v_target = target_vel.copy() if target_vel is not None else np.zeros(3)
    v_target[2] = 0.0

    # 1. Calculate the initial shadow position on the 2D plane
    error_vector = p_boat - p_target
    dist_2d = np.linalg.norm(error_vector[0:2])
    
    if dist_2d < 0.01:
        direction_vect = np.array([-1.0, 0.0, 0.0])
    else:
        direction_vect = error_vector / dist_2d
        
    current_ref_pos = p_target + direction_vect * desired_dist

    # 2. Generate the N-step horizon
    for k in range(int(horizon_N)):
        ref_state = np.zeros(12)
        
        # Predict target future position using constant velocity
        target_pos_future = p_target + v_target * (k * dt)
        
        # Move the shadow position along with the target
        ref_pos_future = current_ref_pos + v_target * (k * dt)
        ref_state[0:3] = ref_pos_future
        
        # Calculate yaw to face the future target position
        pointing_vec = target_pos_future - ref_pos_future
        yaw_des = np.arctan2(pointing_vec[1], pointing_vec[0])
        
        # Unwrap yaw relative to current boat state to prevent 360-degree spins
        diff = yaw_des - boat_state[5]
        if abs(diff) > np.pi:
            if diff > np.pi: yaw_des -= 2 * np.pi
            elif diff < -np.pi: yaw_des += 2 * np.pi
            
        ref_state[5] = yaw_des
        
        # Assign velocities (Strictly forward surge, no sway or heave)
        # We assume the boat matches the target's speed in the direction of the shadow's movement
        speed_ref = np.linalg.norm(v_target[0:2])
        ref_state[6] = speed_ref # u (surge)
        
        # Force strict underactuated boat constraints
        ref_state[4] = 0.0 # pitch
        ref_state[7] = 0.0 # v (sway)
        ref_state[8] = 0.0 # w (heave)

        trajectory.append(ref_state)

    return np.array(trajectory)

def get_dyn_prob_ref(rov_state, kf_est, kf_cov, time_lost, speed=0.8):
    """
    Generate ref based on cov also
    """

    last_pos = kf_est[0:3]
    last_vel = kf_est[3:6]
    predicted_target_pos = last_pos + last_vel*time_lost

    P_pos = kf_cov[0:3]
    P_vel = kf_cov[3:6]
    # uncertainty propagation simplified, bigger sigma where there is uncertainty
    sigma_t = P_pos + (P_vel * (time_lost**2))

    current_pos = rov_state[0:3]
    dist_pred = np.linalg.norm(predicted_target_pos - current_pos)

    ref = np.zeros(12)
    if dist_pred > 5.0:
        direction = (predicted_target_pos - current_pos) / dist_pred
        
        p_ref = current_pos + direction * 1.5
        ref[0:3] = p_ref
        ref[6] = 0.8 

    else:
        
        # Probability cloud obtained with eigendecomposition (autovalori)
        eigvals, eigvecs = np.linalg.eigh(sigma_t[0:2, 0:2])

        i = eigvals.argsort()
        eigvals = eigvals[i]
        eigvecs = eigvecs[:,i]

        # a,b dell'ellisse
        a = 2*np.sqrt(eigvals[1])
        b = 2*np.sqrt(eigvals[0])

        # 0.4 cause dont want the orbit to be too fast?
        theta = time_lost*0.4 

        x_local = a*np.cos(theta)
        y_local = b*np.sin(theta)

        xy_offset = eigvecs @ np.array([x_local, y_local])

        p_ref = predicted_target_pos.copy()
        p_ref[0] += xy_offset[0]
        p_ref[1] += xy_offset[1]
        p_ref[2] = predicted_target_pos[2]

        ref[0:3] = p_ref
        ref[6] = speed

    look_vec = ref[0:3] - current_pos
    yaw_des = np.arctan2(look_vec[1], look_vec[0])
    
    # Yaw Unwrap
    diff = yaw_des - rov_state[5]
    if abs(diff) > np.pi:
        if diff > np.pi: yaw_des -= 2 * np.pi
        elif diff < -np.pi: yaw_des += 2 * np.pi
    
    ref[5] = yaw_des
    
    return ref


def get_dyn_prob_BOAT(boat_state, kf_est, kf_cov, time_lost, speed=0.8):
    """
    Generate probabilistic search reference for a surface vessel.
    """
    last_pos = kf_est[0:3].copy()
    last_vel = kf_est[3:6].copy()
    
    last_pos[2] = 0.0 
    last_vel[2] = 0.0

    predicted_target_pos = last_pos + last_vel * time_lost
    
    P_pos = kf_cov[0:3]
    P_vel = kf_cov[3:6]
    sigma_t = P_pos + (P_vel * (time_lost**2))

    current_pos = boat_state[0:3].copy()
    current_pos[2] = 0.0 
    dist_pred = np.linalg.norm((predicted_target_pos - current_pos)[0:2])

    ref = np.zeros(12)
    if dist_pred > 5.0:
        direction = (predicted_target_pos - current_pos) / dist_pred
        direction[2] = 0.0 # Flatten direction vector
        
        p_ref = current_pos + direction * 1.5
        ref[0:3] = p_ref
        ref[6] = 0.8 
    else:
        eigvals, eigvecs = np.linalg.eigh(sigma_t[0:2, 0:2])

        i = eigvals.argsort()
        eigvals = eigvals[i]
        eigvecs = eigvecs[:,i]

        a = 2 * np.sqrt(np.abs(eigvals[1]))
        b = 2 * np.sqrt(np.abs(eigvals[0]))

        theta = time_lost * 0.4 

        x_local = a * np.cos(theta)
        y_local = b * np.sin(theta)

        xy_offset = eigvecs @ np.array([x_local, y_local])

        p_ref = predicted_target_pos.copy()
        p_ref[0] += xy_offset[0]
        p_ref[1] += xy_offset[1]
        p_ref[2] = 0.0 # Force Z to 0

        ref[0:3] = p_ref
        ref[6] = speed

    look_vec = ref[0:3] - current_pos
    yaw_des = np.arctan2(look_vec[1], look_vec[0])
    
    diff = yaw_des - boat_state[5]
    if abs(diff) > np.pi:
        if diff > np.pi: yaw_des -= 2 * np.pi
        elif diff < -np.pi: yaw_des += 2 * np.pi
    
    ref[5] = yaw_des
    
    return ref