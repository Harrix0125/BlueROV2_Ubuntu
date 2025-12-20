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


def plot_TT_3d(ref_x, ref_y, ref_z, rov_x, rov_y, rov_z, thrust_history, dt):
    """
    ref_x, ref_y, ref_z: 1D arrays of the Target/Reference path
    rov_traj: (N, 12) or (N, 3) array of the actual ROV state history
    thrust_history: (N, 6) or (N, 8) array of thruster values
    dt: Time step in seconds (for the X-axis of the thruster plot)
    """
    fig = plt.figure(figsize=(16, 8))
    
    # --- 1. 3D Trajectory Plot ---
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot Reference (The path we wanted to follow) - Dotted Line
    ax1.plot(ref_x, ref_y, ref_z, label='Reference (Target)', 
             linestyle='--', color='gray', alpha=0.7)
    
    # Plot Actual (The path the ROV took) - Solid Line
    ax1.plot(rov_x, rov_y, rov_z, label='ROV Trajectory', 
             linewidth=2, color='blue')
    
    # Start/End Markers
    ax1.scatter(rov_x[0], rov_y[0], rov_z[0], c='green', marker='o', s=50, label='Start')
    ax1.scatter(rov_x[-1], rov_y[-1], rov_z[-1], c='red', marker='x', s=50, label='End')

    # Formatting
    ax1.set_title('3D Path (NED Frame)')
    ax1.set_xlabel('North [m]')
    ax1.set_ylabel('East [m]')
    ax1.set_zlabel('Down [m]')
    ax1.invert_zaxis() # Depth is positive in NED
    
    # --- Aspect Ratio Fix (Robust) ---
    # We combine both paths to find the true max/min of the scene
    all_x = np.concatenate((ref_x, rov_x))
    all_y = np.concatenate((ref_y, rov_y))
    all_z = np.concatenate((ref_z, rov_z))
    
    max_range = np.array([all_x.max()-all_x.min(), 
                          all_y.max()-all_y.min(), 
                          all_z.max()-all_z.min()]).max() / 2.0

    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5

    # Force the limits to be a cube centered on the data
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set the box aspect to be 1:1:1 so it doesn't look distorted
    ax1.set_box_aspect((1, 1, 1))
    
    ax1.legend()
    
    # --- 2. Thruster Plot ---
    ax2 = fig.add_subplot(122)
    
    # Create time axis based on the number of steps and dt
    time_axis = np.arange(thrust_history.shape[0]) * dt
    
    # Loop through each thruster (column)
    for t in range(thrust_history.shape[1]):
        ax2.plot(time_axis, thrust_history[:, t], label=f'T{t+1}')
        
    ax2.set_title('Thruster Outputs')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Force / PWM')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Legend outside the plot
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()