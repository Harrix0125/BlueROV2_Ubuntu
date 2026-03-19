import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration
# ==========================================
# Make sure to put the actual path to your .npz file here
file_path = 'BB_home_test.npz' 

# Load the data
data = np.load(file_path)

# ==========================================
# 2. Data Extraction
# ==========================================
# Using the exact keys found in your archive
x_act = data['rov_x']
y_act = data['rov_y']
x_ref = data['ref_x']
y_ref = data['ref_y']
thrusters = data['thrust']

# Handle the time axis using 'dt'
dt_data = data['dt']

# If dt is a single scalar value (e.g., 0.1), generate a time array based on the length of the run
if dt_data.size == 1:
    time = np.arange(len(x_act)) * float(dt_data)
# If dt is an array of timestamps or deltas of the same length
elif len(dt_data) == len(x_act):
    time = dt_data 
# Fallback to simple step counts if there's a shape mismatch
else:
    time = np.arange(len(x_act))

# ==========================================
# 3. Plotting
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: X, Y Position vs Reference ---
ax1.plot(y_ref, x_ref, 'k--', label='Reference Path', linewidth=2)
ax1.plot(y_act, x_act, 'b-', label='Actual Position', linewidth=2)

# Mark start and end points
ax1.scatter(y_ref[0], x_ref[0], color='green', marker='o', s=100, label='Start')
ax1.scatter(y_ref[-1], x_ref[-1], color='red', marker='x', s=100, label='End')

ax1.set_title('Trajectory: Actual vs Reference')
ax1.set_xlabel('East / Y')
ax1.set_ylabel('North / X')
ax1.legend()
ax1.grid(True)
ax1.axis('equal') # Keeps the aspect ratio 1:1

# --- Plot 2: Thruster Inputs vs Time ---
# Check if thrusters is a 2D array (multiple thrusters) or 1D (single combined value)
if thrusters.ndim > 1:
    num_thrusters = thrusters.shape[1]
    for i in range(num_thrusters):
        ax2.plot(time, thrusters[:, i], label=f'Thruster {i+1}')
else:
    ax2.plot(time, thrusters, label='Thruster Output')

ax2.set_title('MPC Thruster Output Over Time')
ax2.set_xlabel('Time (seconds or steps)')
ax2.set_ylabel('Commanded Thrust')
ax2.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
ax2.grid(True)

# Render the plots
plt.tight_layout()
plt.show()