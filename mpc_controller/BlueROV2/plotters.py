import matplotlib.pyplot as plt
import numpy as np

def plot_TT_3d(target_x, target_y, target_z,
               ref_x, ref_y, ref_z, 
               rov_x, rov_y, rov_z, 
               rov_roll, rov_pitch, rov_yaw,
               thrust_history, dt, 
               arrow_stride=20, arrow_length=1.5): 
    """
    ref_x, ref_y, ref_z: 1D arrays of the Target/Reference path
    rov_x, ... rov_yaw: 1D arrays of Actual ROV state (Angles in Radians!)
    thrust_history: (N, 6) or (N, 8) array of thruster values
    dt: Time step
    arrow_stride: Plot an arrow every N steps (to avoid clutter)
    arrow_length: Visual length of the direction arrows
    """
    fig = plt.figure(figsize=(16, 8))
    
    # --- 1. 3D Trajectory Plot ---
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot Target
    ax1.plot(target_x, target_y, target_z, label='Target', 
             linewidth=2, color='green')
    
    # Plot Reference
    ax1.plot(ref_x, ref_y, ref_z, label='Reference', 
             linestyle='--', color='gray', alpha=0.7)
    
    # Plot Actual Trajectory
    ax1.plot(rov_x, rov_y, rov_z, label='ROV Trajectory', 
             linewidth=2, color='blue')
    
    # --- NEW: Orientation Vectors (Quivers) ---
    # We only plot a subset of points to keep the graph readable
    # Create indices: start at 0, go to end, step by arrow_stride
    indices = np.arange(0, len(rov_x), arrow_stride)
    
    # Filter data for these indices
    sub_x = rov_x[indices]
    sub_y = rov_y[indices]
    sub_z = rov_z[indices]
    sub_pitch = rov_pitch[indices]
    sub_yaw = rov_yaw[indices]
    
    # Calculate the Direction Vector (Body X-axis) converted to NED coordinates
    # For a standard rotation sequence (Z-Y-X), the Body X-axis in NED is:
    # u (North) = cos(pitch) * cos(yaw)
    # v (East)  = cos(pitch) * sin(yaw)
    # w (Down)  = -sin(pitch)
    
    # Note: Roll does not affect the direction of the X-axis (forward vector), 
    # only the rotation around it, so we don't strictly need it for the arrow direction.
    
    vec_u = np.cos(sub_pitch) * np.cos(sub_yaw)
    vec_v = np.cos(sub_pitch) * np.sin(sub_yaw)
    vec_w = -np.sin(sub_pitch)
    
    # Plot the arrows (Red arrows indicate "Heading")
    ax1.quiver(sub_x, sub_y, sub_z, 
               vec_u, vec_v, vec_w, 
               length=arrow_length, normalize=True, color='red', alpha=0.6, label='Heading')

    # Start/End Markers
    ax1.scatter(rov_x[0], rov_y[0], rov_z[0], c='green', marker='o', s=50, label='Start')
    ax1.scatter(rov_x[-1], rov_y[-1], rov_z[-1], c='red', marker='x', s=50, label='End')

    # Formatting
    ax1.set_title('3D Path with Heading Arrows')
    ax1.set_xlabel('North [m]')
    ax1.set_ylabel('East [m]')
    ax1.set_zlabel('Down [m]')
    ax1.invert_zaxis() # Depth is positive in NED
    
    # --- Aspect Ratio Fix ---
    all_x = np.concatenate((ref_x, rov_x))
    all_y = np.concatenate((ref_y, rov_y))
    all_z = np.concatenate((ref_z, rov_z))
    max_range = np.array([all_x.max()-all_x.min(), 
                          all_y.max()-all_y.min(), 
                          all_z.max()-all_z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (all_x.max()+all_x.min())*0.5, (all_y.max()+all_y.min())*0.5, (all_z.max()+all_z.min())*0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    ax1.set_box_aspect((1, 1, 1))
    ax1.legend()
    
    # --- 2. Thruster Plot ---
    ax2 = fig.add_subplot(122)
    time_axis = np.arange(thrust_history.shape[0]) * dt
    for t in range(thrust_history.shape[1]):
        ax2.plot(time_axis, thrust_history[:, t], label=f'T{t+1}')
        
    ax2.set_title('Thruster Outputs')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Force / PWM')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()


def LOS_plot_dynamics(rov_x, rov_y, rov_z, target_data, dt, desired_dist=2.0):
    """
    Plots Position, Depth, Velocity, and Distance to Target.
    """
    # --- DATA PREP ---
    p_rov = np.array([rov_x, rov_y, rov_z]).T
    p_target = target_data[:len(rov_x), 0:3]
    time = np.arange(len(rov_x)) * dt
    
    # Calculate Distance
    distances = np.linalg.norm(p_rov - p_target, axis=1)
    
    # Calculate Velocity (Scalar Speed)
    # Gradient gives change per step; divide by dt for per second
    # axis=0 is time dimension
    vel_vector = np.gradient(p_rov, axis=0) / dt 
    speed = np.linalg.norm(vel_vector, axis=1)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Top-Down View
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(p_target[:, 0], p_target[:, 1], 'r--', label='Target')
    ax1.plot(p_rov[:, 0], p_rov[:, 1], 'b-', linewidth=2, label='ROV')
    # Standoff Ring
    final_t = p_target[-1]
    circle = plt.Circle((final_t[0], final_t[1]), desired_dist, color='r', fill=False, linestyle=':', label='2m Ring')
    ax1.add_patch(circle)
    ax1.set_title("1. Top-Down Path (X-Y)")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()

    # 2. Depth View
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time, p_target[:, 2], 'r--', label='Target Depth')
    ax2.plot(time, p_rov[:, 2], 'b-', label='ROV Depth')
    #ax2.invert_yaxis() # Depth is usually positive down, so we invert to make "surface" at top
    ax2.set_title("2. Depth Profile (Z)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Depth [m]")
    ax2.grid(True)
    ax2.legend()

    # 3. VELOCITY GRAPH (New)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(time, speed, 'm-', linewidth=2, label='ROV Speed')
    ax3.set_title("3. ROV Speed over Time")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Speed [m/s]")
    ax3.grid(True)
    ax3.legend()

    # 4. Distance Graph (Moved here)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(time, distances, 'k-', linewidth=2, label='Actual Dist')
    ax4.axhline(desired_dist, color='r', linestyle='--', label=f'Desired ({desired_dist}m)')
    
    # Color bands for error
    ax4.fill_between(time, distances, desired_dist, where=(distances > desired_dist), color='red', alpha=0.1)
    ax4.fill_between(time, distances, desired_dist, where=(distances < desired_dist), color='green', alpha=0.1)
    
    ax4.set_title("4. Distance to Target")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Distance [m]")
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()


def LOS_plot_camera_fov(rov_x, rov_y, rov_z, rov_psi, rov_pitch, target_data, dt, cam_fov_h=90, cam_fov_v=80):
    """
    Plots whether the target is visible in the Camera (Horizontal and Vertical).
    
    Args:
        rov_pitch: History of ROV Pitch angles (radians). Pass zeros if unknown.
        cam_fov_h: Camera Horizontal Field of View (degrees).
        cam_fov_v: Camera Vertical Field of View (degrees).
    """
    n_steps = len(rov_x)
    p_rov = np.array([rov_x, rov_y, rov_z]).T
    p_target = target_data[:n_steps, 0:3]
    psi_rov = np.array(rov_psi)
    theta_rov = np.array(rov_pitch)
    time = np.arange(n_steps) * dt

    # --- 1. HORIZONTAL (Yaw) CALCULATION ---
    delta_pos = p_target - p_rov
    # Angle of the vector connecting ROV to Target (XY plane)
    bearing_xy = np.arctan2(delta_pos[:, 1], delta_pos[:, 0])
    
    # Error = Bearing - Heading
    err_yaw = bearing_xy - psi_rov
    # Wrap to -pi ... +pi
    err_yaw = (err_yaw + np.pi) % (2 * np.pi) - np.pi
    err_yaw_deg = np.degrees(err_yaw)

    # --- 2. VERTICAL (Pitch) CALCULATION (ZX Plane) ---
    # Horizontal distance to target
    dist_xy = np.linalg.norm(delta_pos[:, 0:2], axis=1)
    # Vertical distance (Standard NED: Z is positive DOWN)
    # If Target Z > ROV Z, target is DEEPER.
    delta_z = p_target[:, 2] - p_rov[:, 2]
    
    # Angle to target (Elevation). 
    # Positive = Target is DEEPER (Looking down). 
    # Negative = Target is SHALLOWER (Looking up).
    bearing_vert = np.arctan2(delta_z, dist_xy)
    
    # Pitch Convention: Usually Positive Pitch = Nose UP.
    # Therefore, if we Pitch UP (Pos), we look at shallower things (Neg Z).
    # We need to be careful with signs here.
    # Let's define "Camera Vertical Angle" relative to horizon:
    # Camera_Angle = -Theta (because Theta is Nose Up, but +Z is Down)
    # This gets confusing. The simplest check:
    # Where am I looking? -> Pitch (Theta)
    # Where should I look? -> -bearing_vert (if using standard Pitch=Up / Z=Down)
    
    # SIMPLIFIED LOGIC:
    # If I am perfectly flat (Pitch=0), I see straight ahead.
    # If Target is deeper (+Z), bearing_vert is positive. 
    # To see it, I must Pitch DOWN (Negative Theta).
    # So Error = bearing_vert - (-Theta) ... or simpler:
    # Total Vertical Angle of Target relative to Camera Center = bearing_vert + Theta
    # (Checking: If Target is 45 deg down (+45), and I pitch 45 deg down (-45), Sum = 0. Perfect.)
    
    err_pitch_rad = bearing_vert + theta_rov
    err_pitch_deg = np.degrees(err_pitch_rad)

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # PLOT 1: Horizontal Visibility (Yaw)
    ax1.plot(time, err_yaw_deg, 'k-', label='Horizontal Error')
    ax1.axhline(0, color='g', linestyle='--', alpha=0.5)
    
    # FOV Bands
    h_lim = cam_fov_h / 2
    ax1.fill_between(time, -h_lim, h_lim, color='green', alpha=0.1, label=f'Cam FOV (+/-{h_lim}°)')
    
    ax1.set_title("Horizontal Visibility (Yaw Error)")
    ax1.set_ylabel("Error [deg] (Left/Right)")
    ax1.set_ylim(-180, 180)
    ax1.grid(True)
    ax1.legend(loc='upper right')

    # PLOT 2: Vertical Visibility (Pitch)
    ax2.plot(time, err_pitch_deg, 'b-', label='Vertical Error')
    ax2.axhline(0, color='g', linestyle='--', alpha=0.5)
    
    # FOV Bands
    v_lim = cam_fov_v / 2
    ax2.fill_between(time, -v_lim, v_lim, color='orange', alpha=0.1, label=f'Cam FOV (+/-{v_lim}°)')
    
    ax2.set_title("Vertical Visibility (Pitch/Depth Error)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Error [deg] (Up/Down)")
    ax2.set_ylim(-90, 90) # Vertical angles rarely exceed 90
    ax2.grid(True)
    ax2.legend(loc='upper right')

    plt.tight_layout()

