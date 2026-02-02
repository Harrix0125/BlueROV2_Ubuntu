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
    ref_state_t0 = get_shadow_ref(rov_state, target_state, target_vel, desired_dist)
    
    p_ref_current = ref_state_t0[0:3]
    v_global_propagation = target_vel if target_vel is not None else np.zeros(3)

    trajectory = []

    # Loop to generate future points assuming constant velocity
    for k in range(int(horizon_N)):  # NOTE: I removed the /5 division to match standard NMPC logic, add it back if intended
        time_offset = k * dt
        p_ref_future = p_ref_current + v_global_propagation * time_offset
        
        ref_state_k = ref_state_t0.copy()
        ref_state_k[0:3] = p_ref_future
        trajectory.append(ref_state_k)

    return np.array(trajectory)