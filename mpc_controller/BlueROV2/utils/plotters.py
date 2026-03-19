import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
def plot_TT_3d(target_x, target_y, target_z,
               ref_x, ref_y, ref_z, 
               rov_x, rov_y, rov_z, 
               rov_roll, rov_pitch, rov_yaw,
               thrust_history, dt, 
               arrow_stride=20, arrow_length=0.0): 
    
    # --- FIGURE 1: 3D Trajectory Plot ---
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')

    ax1.plot(target_x, target_y, target_z, label='Target', linewidth=2, color='green')
    ax1.plot(ref_x, ref_y, ref_z, label='Reference', linestyle='--', color='gray', alpha=0.7)
    ax1.plot(rov_x, rov_y, rov_z, label='ROV Trajectory', linewidth=2, color='blue')
    
    indices = np.arange(0, len(rov_x), arrow_stride)
    sub_x = rov_x[indices]
    sub_y = rov_y[indices]
    sub_z = rov_z[indices]
    sub_pitch = rov_pitch[indices]
    sub_yaw = rov_yaw[indices]
    
    vec_u = np.cos(sub_pitch) * np.cos(sub_yaw)
    vec_v = np.cos(sub_pitch) * np.sin(sub_yaw)
    vec_w = -np.sin(sub_pitch)
    
    ax1.scatter(rov_x[0], rov_y[0], rov_z[0], c='green', marker='o', s=50, label='Start')
    ax1.scatter(rov_x[-1], rov_y[-1], rov_z[-1], c='red', marker='x', s=50, label='End')

    ax1.set_title('3D Path with Heading Arrows')
    ax1.set_xlabel('North [m]')
    ax1.set_ylabel('East [m]')
    ax1.set_zlabel('Down [m]')
    ax1.invert_zaxis() 
    
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
    fig1.tight_layout()
    
    # --- FIGURE 2: Thruster Plots ---
    fig2 = plt.figure(figsize=(10, 8))
    time_axis = np.arange(thrust_history.shape[0]) * dt
    num_thrusters = thrust_history.shape[1]

    if num_thrusters >= 6:
        # ROV Configuration: Split Planar (0-3) and Vertical (4+)
        ax_planar = fig2.add_subplot(211)
        for t in range(4):
            ax_planar.plot(time_axis, thrust_history[:, t], label=f'T{t+1} (Planar)')
        ax_planar.set_title('Planar Thrusters Output')
        ax_planar.set_ylabel('Force / PWM')
        ax_planar.grid(True, linestyle='--', alpha=0.5)
        ax_planar.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        ax_vert = fig2.add_subplot(212, sharex=ax_planar)
        for t in range(4, num_thrusters):
            ax_vert.plot(time_axis, thrust_history[:, t], label=f'T{t+1} (Vertical)')
        ax_vert.set_title('Vertical Thrusters Output')
        ax_vert.set_xlabel('Time (s)')
        ax_vert.set_ylabel('Force / PWM')
        ax_vert.grid(True, linestyle='--', alpha=0.5)
        ax_vert.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        # Boat Configuration
        ax_boat = fig2.add_subplot(111)
        for t in range(num_thrusters):
            ax_boat.plot(time_axis, thrust_history[:, t], label=f'T{t+1}')
        ax_boat.set_title('Thruster Outputs')
        ax_boat.set_xlabel('Time (s)')
        ax_boat.set_ylabel('Force / PWM')
        ax_boat.grid(True, linestyle='--', alpha=0.5)
        ax_boat.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    fig2.tight_layout()

    # --- FIGURE 3: Distance From Reference ---
    fig3 = plt.figure(figsize=(10, 4))
    ax_err = fig3.add_subplot(111)
    
    # Calculate Euclidean distance between the vehicle and the reference path
    error_from_ref = np.sqrt((rov_x - ref_x)**2 + (rov_y - ref_y)**2 + (rov_z - ref_z)**2)
    
    ax_err.plot(time_axis, error_from_ref, 'k-', linewidth=2, label='Total Tracking Error')
    ax_err.set_title("Distance from Reference over Time")
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("Error [m]")
    ax_err.grid(True)
    ax_err.legend()
    
    fig3.tight_layout()


def LOS_plot_dynamics(rov_x, rov_y, rov_z, target_data, dt, desired_dist=2.0):
    p_rov = np.array([rov_x, rov_y, rov_z]).T
    p_target = target_data[:len(rov_x), 0:3]
    time = np.arange(len(rov_x)) * dt
    
    distances = np.linalg.norm(p_rov - p_target, axis=1)
    vel_vector = np.gradient(p_rov, axis=0) / dt 
    speed = np.linalg.norm(vel_vector, axis=1)

    # Infer vehicle: If Z is basically 0, it's a surface vessel
    is_boat = np.allclose(rov_z, 0, atol=1e-2)

    # --- FIGURE 1: Top-Down and (Optionally) Depth ---
    if is_boat:
        fig1 = plt.figure(figsize=(7, 5))
        ax1 = fig1.add_subplot(1, 1, 1)
    else:
        fig1 = plt.figure(figsize=(14, 5))
        ax1 = fig1.add_subplot(1, 2, 1)
        
    ax1.plot(p_target[:, 0], p_target[:, 1], 'r--', label='Target')
    ax1.plot(p_rov[:, 0], p_rov[:, 1], 'b-', linewidth=2, label='ROV')
    final_t = p_target[-1]
    circle = plt.Circle((final_t[0], final_t[1]), desired_dist, color='r', fill=False, linestyle=':', label='2m Ring')
    ax1.add_patch(circle)
    ax1.set_title("1. Top-Down Path (X-Y)")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()

    if not is_boat:
        ax2 = fig1.add_subplot(1, 2, 2)
        ax2.plot(time, p_target[:, 2], 'r--', label='Target Depth')
        ax2.plot(time, p_rov[:, 2], 'b-', label='ROV Depth')
        ax2.set_title("2. Depth Profile (Z)")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Depth [m]")
        ax2.grid(True)
        ax2.legend()
        
    fig1.tight_layout()

    # --- FIGURE 2: Speed and Distance ---
    fig2 = plt.figure(figsize=(14, 5))
    
    ax3 = fig2.add_subplot(1, 2, 1)
    ax3.plot(time, speed, 'm-', linewidth=2, label='Vehicle Speed')
    ax3.set_title("3. Vehicle Speed over Time")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Speed [m/s]")
    ax3.grid(True)
    ax3.legend()

    ax4 = fig2.add_subplot(1, 2, 2)
    ax4.plot(time, distances, 'k-', linewidth=2, label='Actual Dist')
    ax4.axhline(desired_dist, color='r', linestyle='--', label=f'Desired ({desired_dist}m)')
    ax4.fill_between(time, distances, desired_dist, where=(distances > desired_dist), color='red', alpha=0.1)
    ax4.fill_between(time, distances, desired_dist, where=(distances < desired_dist), color='green', alpha=0.1)
    ax4.set_title("4. Distance to Target")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Distance [m]")
    ax4.grid(True)
    ax4.legend()
    fig2.tight_layout()

def LOS_plot_dynamics_desired(rov_x, rov_y, rov_z, target_data, dt, desired_dist=2.0, ref_x=None, ref_y=None, ref_z=None):
    p_rov = np.array([rov_x, rov_y, rov_z]).T
    p_target = target_data[:len(rov_x), 0:3]
    time = np.arange(len(rov_x)) * dt
    
    distances = np.linalg.norm(p_rov - p_target, axis=1)
    vel_vector = np.gradient(p_rov, axis=0) / dt 
    speed = np.linalg.norm(vel_vector, axis=1)

    # Infer vehicle: If Z is basically 0, it's a surface vessel
    is_boat = np.allclose(rov_z, 0, atol=1e-2)

    # --- FIGURE 1: Top-Down and (Optionally) Depth ---
    if is_boat:
        fig1 = plt.figure(figsize=(7, 5))
        ax1 = fig1.add_subplot(1, 1, 1)
    else:
        fig1 = plt.figure(figsize=(14, 5))
        ax1 = fig1.add_subplot(1, 2, 1)
        
    ax1.plot(p_target[:, 0], p_target[:, 1], 'r--', label='Target')
    ax1.plot(p_rov[:, 0], p_rov[:, 1], 'b-', linewidth=2, label='ROV Actual')
    
    # NEW: Plot the desired reference path if provided
    if ref_x is not None and ref_y is not None:
        ax1.plot(ref_x, ref_y, 'g:', linewidth=2, alpha=0.8, label='Desired Path')

    final_t = p_target[-1]
    circle = plt.Circle((final_t[0], final_t[1]), desired_dist, color='r', fill=False, linestyle=':', label='2m Ring')
    ax1.add_patch(circle)
    ax1.set_title("1. Top-Down Path (X-Y)")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()

    if not is_boat:
        ax2 = fig1.add_subplot(1, 2, 2)
        ax2.plot(time, p_target[:, 2], 'r--', label='Target Depth')
        ax2.plot(time, p_rov[:, 2], 'b-', label='ROV Actual Depth')
        
        # NEW: Plot the desired reference depth if provided
        if ref_z is not None:
            ax2.plot(time, ref_z, 'g:', linewidth=2, alpha=0.8, label='Desired Depth')
            
        ax2.set_title("2. Depth Profile (Z)")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Depth [m]")
        ax2.grid(True)
        ax2.legend()
        
    fig1.tight_layout()

    # --- FIGURE 2: Speed and Distance ---
    fig2 = plt.figure(figsize=(14, 5))
    
    ax3 = fig2.add_subplot(1, 2, 1)
    ax3.plot(time, speed, 'm-', linewidth=2, label='Vehicle Speed')
    ax3.set_title("3. Vehicle Speed over Time")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Speed [m/s]")
    ax3.grid(True)
    ax3.legend()

    ax4 = fig2.add_subplot(1, 2, 2)
    ax4.plot(time, distances, 'k-', linewidth=2, label='Actual Dist')
    ax4.axhline(desired_dist, color='r', linestyle='--', label=f'Desired ({desired_dist}m)')
    ax4.fill_between(time, distances, desired_dist, where=(distances > desired_dist), color='red', alpha=0.1)
    ax4.fill_between(time, distances, desired_dist, where=(distances < desired_dist), color='green', alpha=0.1)
    ax4.set_title("4. Distance to Target")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Distance [m]")
    ax4.grid(True)
    ax4.legend()
    fig2.tight_layout()

def LOS_plot_camera_fov(rov_x, rov_y, rov_z, rov_psi, rov_pitch, target_data, dt, cam_fov_h=90, cam_fov_v=80):
    n_steps = len(rov_x)
    p_rov = np.array([rov_x, rov_y, rov_z]).T
    p_target = target_data[:n_steps, 0:3]
    psi_rov = np.array(rov_psi)
    theta_rov = np.array(rov_pitch)
    time = np.arange(n_steps) * dt

    # --- HORIZONTAL ---
    delta_pos = p_target - p_rov
    bearing_xy = np.arctan2(delta_pos[:, 1], delta_pos[:, 0])
    err_yaw = bearing_xy - psi_rov
    err_yaw = (err_yaw + np.pi) % (2 * np.pi) - np.pi
    err_yaw_deg = np.degrees(err_yaw)

    # --- VERTICAL ---
    dist_xy = np.linalg.norm(delta_pos[:, 0:2], axis=1)
    delta_z = p_target[:, 2] - p_rov[:, 2]
    bearing_vert = np.arctan2(delta_z, dist_xy)
    err_pitch_rad = bearing_vert + theta_rov
    err_pitch_deg = np.degrees(err_pitch_rad)

    # Infer vehicle: If Z and Pitch are basically 0, it's the BlueBoat
    is_boat = np.allclose(rov_z, 0, atol=1e-2) and np.allclose(rov_pitch, 0, atol=1e-2)

    if is_boat:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        
        ax1.plot(time, err_yaw_deg, 'k-', label='Horizontal Error')
        ax1.axhline(0, color='g', linestyle='--', alpha=0.5)
        h_lim = cam_fov_h / 2
        ax1.fill_between(time, -h_lim, h_lim, color='green', alpha=0.1, label=f'Cam FOV (+/-{h_lim}°)')
        ax1.set_title("Horizontal Visibility (Yaw Error) - BlueBoat")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Error [deg] (Left/Right)")
        ax1.set_ylim(-180, 180)
        ax1.grid(True)
        ax1.legend(loc='upper right')
        
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        ax1.plot(time, err_yaw_deg, 'k-', label='Horizontal Error')
        ax1.axhline(0, color='g', linestyle='--', alpha=0.5)
        h_lim = cam_fov_h / 2
        ax1.fill_between(time, -h_lim, h_lim, color='green', alpha=0.1, label=f'Cam FOV (+/-{h_lim}°)')
        ax1.set_title("Horizontal Visibility (Yaw Error)")
        ax1.set_ylabel("Error [deg] (Left/Right)")
        ax1.set_ylim(-180, 180)
        ax1.grid(True)
        ax1.legend(loc='upper right')

        ax2.plot(time, err_pitch_deg, 'b-', label='Vertical Error')
        ax2.axhline(0, color='g', linestyle='--', alpha=0.5)
        v_lim = cam_fov_v / 2
        ax2.fill_between(time, -v_lim, v_lim, color='orange', alpha=0.1, label=f'Cam FOV (+/-{v_lim}°)')
        ax2.set_title("Vertical Visibility (Pitch/Depth Error)")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Error [deg] (Up/Down)")
        ax2.set_ylim(-90, 90) 
        ax2.grid(True)
        ax2.legend(loc='upper right')

    plt.tight_layout()

def LOS_interactive_viewer(rov_x, rov_y, rov_z, target_data, dt):
    """
    Creates an interactive plot with a time slider to visualize 
    ROV and Target positions at specific timestamps.
    """
    # --- 1. DATA PREP ---
    p_rov = np.array([rov_x, rov_y, rov_z]).T
    p_target = target_data[:len(rov_x), 0:3]
    time = np.arange(len(rov_x)) * dt
    max_time = time[-1]

    # --- 2. SETUP FIGURE AND AXES ---
    # We leave extra space at the bottom for the slider
    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25) # Make room at bottom

    # Subplot 1: Top-Down View (X-Y)
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_xy.set_title("Top-Down Trajectory (X-Y)")
    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.axis('equal')
    ax_xy.grid(True)

    # Subplot 2: Depth View (Time-Z) or Side View
    # Let's do Side View (X-Z) so it looks like "Space" 
    # (Or stick to Time-Depth if you prefer)
    ax_z = fig.add_subplot(1, 2, 2)
    ax_z.set_title("Depth Profile (Time vs Z)")
    ax_z.set_xlabel("Time [s]")
    ax_z.set_ylabel("Depth [m]")
    ax_z.invert_yaxis() # Surface at top
    ax_z.grid(True)

    # --- 3. PLOT STATIC BACKGROUND (The full history) ---
    # These lines won't move. They show the path taken.
    ax_xy.plot(p_target[:, 0], p_target[:, 1], 'r--', alpha=0.3, label='Target Path')
    ax_xy.plot(p_rov[:, 0], p_rov[:, 1], 'b-', alpha=0.3, label='ROV Path')
    
    ax_z.plot(time, p_target[:, 2], 'r--', alpha=0.3)
    ax_z.plot(time, p_rov[:, 2], 'b-', alpha=0.3)

    # --- 4. INITIALIZE MOVING POINTS (The "Dots") ---
    # We start at index 0 (t=0)
    # comma after variable name extracts the object from the list
    rov_dot_xy, = ax_xy.plot([], [], 'bo', markersize=10, label='ROV Current')
    tgt_dot_xy, = ax_xy.plot([], [], 'ro', markersize=10, label='Target Current')
    
    rov_dot_z, = ax_z.plot([], [], 'bo', markersize=10)
    tgt_dot_z, = ax_z.plot([], [], 'ro', markersize=10)

    ax_xy.legend(loc='upper right')

    # --- 5. CREATE THE SLIDER ---
    # Define the axis area where the slider will live [left, bottom, width, height]
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    
    time_slider = Slider(
        ax=ax_slider,
        label='Time [s]',
        valmin=0,
        valmax=max_time,
        valinit=0,
        valstep=dt # Snap to your exact time steps
    )

    # --- 6. THE UPDATE FUNCTION ---
    # This runs every time you move the slider
    def update(val):
        current_time = time_slider.val
        # Find the array index closest to the slider time
        idx = int(current_time / dt)
        
        # Ensure we don't go out of bounds
        if idx >= len(rov_x): 
            idx = len(rov_x) - 1

        # UPDATE X-Y DOTS
        rov_dot_xy.set_data([p_rov[idx, 0]], [p_rov[idx, 1]])
        tgt_dot_xy.set_data([p_target[idx, 0]], [p_target[idx, 1]])
        
        # UPDATE DEPTH DOTS (Time vs Depth)
        # Note: For set_data, we need lists/arrays, hence the brackets []
        rov_dot_z.set_data([time[idx]], [p_rov[idx, 2]])
        tgt_dot_z.set_data([time[idx]], [p_target[idx, 2]])
        
        # Redraw the canvas
        fig.canvas.draw_idle()

    # Register the update function with the slider
    time_slider.on_changed(update)
    
    # Run update once to set initial positions
    update(0)

    # Return the slider object to prevent Garbage Collection
    return time_slider, fig 

# --- MAIN EXECUTION ---

# IMPORTANT: You must assign the return value to a variable!
# If you just call the function without 'slider = ...', the slider will freeze.
# slider_obj, fig_obj = LOS_interactive_viewer(rov_x, rov_y, rov_z, target_data, dt)

# plt.show()