import matplotlib.pyplot as plt
import numpy as np
from nmpc_params import NMPC_params as MPCC


def plot_double_target_3d(traj_x, traj_y, traj_z, state_target_1, state_target_2,thrust_history):
    fig = plt.figure(figsize=(15, 7))
    # --- Trajectory Plot ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(traj_x, traj_y, traj_z, label='Path', linewidth=2)
    
    # Add Start and End markers to see the full context
    ax1.scatter(traj_x[0], traj_y[0], traj_z[0], c='g', marker='o', s=50, label='Start')
    ax1.scatter(state_target_1[0], state_target_1[1], state_target_1[2], c='b', marker='x', s=50, label='Target')
    ax1.scatter(state_target_2[0], state_target_2[1], state_target_2[2], c='r', marker='x', s=50, label='Target 2')
    
    ax1.set_title('3D Path (NED Frame)')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.invert_zaxis() # standard for underwater vehicles (Depth is positive)
    
    # 2. FIX 3D ASPECT RATIO (Crucial for realism)
    # This forces the 3D box to have equal visual dimensions
    xyz_limits = np.array([ax1.get_xlim(), ax1.get_ylim(), ax1.get_zlim()])
    xyz_range = np.ptp(xyz_limits, axis=1)
    ax1.set_box_aspect(xyz_range) 
    
    ax1.legend()
    
    # --- Thruster Plot ---
    ax2 = fig.add_subplot(122)
    time_axis = np.arange(thrust_history.shape[0]) * MPCC.T_s
    
    for t in range(thrust_history.shape[1]):
        ax2.plot(time_axis, thrust_history[:, t], label=f'T{t+1}')
        
    ax2.set_title('Thruster Forces')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Force (N)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. MOVE LEGEND OUTSIDE
    # This places the legend outside the box so it doesn't cover the data
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust layout to make room for the external legend
    plt.tight_layout()
    plt.show()

def plot_TT_3d(ref_x, ref_y, ref_z, 
               rov_x, rov_y, rov_z, 
               rov_roll, rov_pitch, rov_yaw,  # <--- NEW INPUTS (Radians)
               thrust_history, dt, 
               arrow_stride=20, arrow_length=0.5): # <--- NEW VISUAL SETTINGS
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
    plt.show()


def LOS_plot_standoff_tracking(rov_x, rov_y, rov_z, target_data, dt, desired_dist=2.0):
    """
    Plots the results of Standoff Tracking.
    
    Args:
        rov_x, rov_y, rov_z: Lists or arrays of ROV position history.
        target_data: Numpy array (N, 12) containing target state history.
        dt: Time step (seconds).
        desired_dist: The target separation distance (default 2.0m).
    """
    
    # 1. Convert ROV lists to numpy arrays for easier math
    p_rov = np.array([rov_x, rov_y, rov_z]).T # Shape (N, 3)
    p_target = target_data[:len(rov_x), 0:3]  # Shape (N, 3) - Match length
    
    # 2. Calculate Distance Error over time
    # Euclidean distance at every step
    distances = np.linalg.norm(p_rov - p_target, axis=1)
    
    # Time vector
    time = np.arange(len(rov_x)) * dt

    # --- PLOTTING ---
    fig = plt.figure(figsize=(14, 8))
    
    # SUBPLOT 1: Top-Down View (2D Plane)
    # This is best to see if you are "following" or "orbiting"
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(p_target[:, 0], p_target[:, 1], 'r--', label='Target (Diver)')
    ax1.plot(p_rov[:, 0], p_rov[:, 1], 'b-', linewidth=2, label='BlueROV2')
    
    # Draw a circle around the LAST target position to show the standoff ring
    final_t = p_target[-1]
    circle = plt.Circle((final_t[0], final_t[1]), desired_dist, color='r', fill=False, linestyle=':', alpha=0.5, label='2m Ring')
    ax1.add_patch(circle)
    
    ax1.set_title("Top-Down View (X-Y)")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.axis('equal') # Important to see true geometry
    ax1.grid(True)
    ax1.legend()

    # SUBPLOT 2: Depth View (Side profile)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time, p_target[:, 2], 'r--', label='Target Depth')
    ax2.plot(time, p_rov[:, 2], 'b-', label='ROV Depth')
    ax2.set_title("Depth Tracking (Z)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Depth [m]")
    ax2.grid(True)
    ax2.legend()

    # SUBPLOT 3: Distance Error (The Critical Metric)
    ax3 = fig.add_subplot(2, 1, 2) # Spans bottom row
    ax3.plot(time, distances, 'k-', linewidth=2, label='Actual Distance')
    ax3.axhline(y=desired_dist, color='r', linestyle='--', linewidth=2, label=f'Desired ({desired_dist}m)')
    
    # Add a fill to show error
    ax3.fill_between(time, distances, desired_dist, where=(distances > desired_dist), color='red', alpha=0.1, interpolate=True)
    ax3.fill_between(time, distances, desired_dist, where=(distances < desired_dist), color='green', alpha=0.1, interpolate=True)
    
    ax3.set_title("Separation Distance vs. Time")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Distance [m]")
    ax3.set_ylim(0, max(np.max(distances), desired_dist) + 1.0)
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.show()

def LOS_plot_heading_error(rov_x, rov_y, rov_psi, target_data, dt):
    """
    Plots the difference between ROV Heading and the geometrical Bearing to the Target.
    
    Args:
        rov_x, rov_y: ROV position lists/arrays
        rov_psi: ROV Yaw angle history (radians)
        target_data: Target state history (N, 12)
        dt: Time step
    """
    # 1. Prepare Data
    n_steps = len(rov_x)
    p_rov = np.array([rov_x, rov_y]).T           # (N, 2)
    p_target = target_data[:n_steps, 0:2]        # (N, 2)
    psi_rov = np.array(rov_psi)                  # (N,)

    # 2. Calculate Geometrical Bearing to Target
    # Vector from ROV to Target
    delta = p_target - p_rov
    
    # atan2(y, x) gives angle from -pi to +pi
    bearing_to_target = np.arctan2(delta[:, 1], delta[:, 0])

    # 3. Calculate Error (Difference)
    # error = bearing - heading
    angle_diff = bearing_to_target - psi_rov

    # 4. WRAP ANGLES (Critical Step)
    # We must ensure the error is between -pi and +pi 
    # (e.g., if bearing is 3.14 and heading is -3.14, error should be 0, not 6.28)
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

    # Convert to Degrees for easier reading
    t = np.arange(n_steps) * dt
    diff_deg = np.degrees(angle_diff)
    
    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Subplot 1: Absolute Angles (Heading vs Bearing)
    # We allow these to wrap or not, depending on preference. 
    # For clarity here, we plot them as-is (radians or degrees)
    ax1.plot(t, np.degrees(psi_rov), 'b-', label='ROV Heading (Psi)')
    ax1.plot(t, np.degrees(bearing_to_target), 'r--', label='Target Bearing')
    ax1.set_title("Heading vs. Target Bearing")
    ax1.set_ylabel("Angle [deg]")
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: The Difference (Error)
    ax2.plot(t, diff_deg, 'k-', linewidth=2, label='Angle Error')
    
    # Add "Zero Error" line
    ax2.axhline(0, color='g', linestyle='--', alpha=0.5)
    
    # Add "Field of View" (FOV) bands 
    # e.g., if your camera has 90deg FOV, +/- 45deg is visible
    ax2.fill_between(t, -45, 45, color='green', alpha=0.1, label='Camera FOV (+/- 45)')

    ax2.set_title("Angle Difference (Where am I looking relative to Target?)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Error [deg]")
    ax2.set_ylim(-180, 180) # Lock limits to full rotation
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()