import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

from nmpc_solver_acados import Acados_Solver_Wrapper
from nmpc_params import NMPC_params as MPCC
from utils import get_linear_traj, robot_plant_step  

def simulation():
    # Initialize the new Acados solver
    # This will trigger code generation and compilation (takes time once)
    print("Compiling Acados Solver...")
    solver = Acados_Solver_Wrapper()
    print("Compilation Complete.")

    state_now = np.zeros(12)
    state_now[2] = -0.5

    state_target = np.zeros(12)

    state_target_1 = np.zeros(12)
    state_target_1[0] = 2.0
    state_target_1[1] = -2.0
    state_target_1[2] = -5
    state_target_1[5] = 0

    state_target_2 = np.zeros(12)
    state_target_2[0] = -4.0
    state_target_2[1] = -5.0
    state_target_2[2] = -2.0
    state_target_2[5] = 0


    # Storage
    traj_x, traj_y, traj_z = [], [], []
    thrust_history = []   
    t0 = time.time()
    t_simulation = 10 #sec
    steps_tot = int(t_simulation /  MPCC.T_s)
    round = 1
    state_target = state_target_1
    state_moving = get_linear_traj(state_now, state_target_1, steps_tot)
    is_target_moving = False
    
    for i in range(steps_tot): 
        # Solve
        if is_target_moving:
            u_optimal = solver.solve(state_now, state_moving[:,i])
        else:
            u_optimal = solver.solve(state_now, state_target)
        
        # Plant Step
        state_now = robot_plant_step(state_now, u_optimal, MPCC.T_s)

        traj_x.append(state_now[0])
        traj_y.append(state_now[1])
        traj_z.append(state_now[2])
        thrust_history.append(u_optimal)
        
        dist = np.linalg.norm(state_now[:3] - state_target[:3])
        
        if dist < 0.05 and round == 2:
            print(f"Target Reached at step {i}!")
            break
        elif dist < 0.05 and round == 1:
            print(f"First Target Reached at step {i}, setting new target.")
            state_target = state_target_2
            state_moving = get_linear_traj(state_now, state_target_2, steps_tot)
            round += 1

    # 3. Plotting
    t_end = time.time()
    t_sum = t_end - t0
    print(f"{t_sum:.4f}s is the total computational time!")
    
    print("Plotting results...")
    thrust_history = np.array(thrust_history)
    
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

if __name__ == "__main__":
    simulation()