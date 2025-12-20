import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

from nmpc_solver_acados import Acados_Solver_Wrapper
from nmpc_params import NMPC_params as MPCC
from utils import get_linear_traj, robot_plant_step, generate_target_trajectory
from plotters import plot_double_target_3d, plot_TT_3d
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
    t_simulation = 40 #sec
    steps_tot = int(t_simulation /  MPCC.T_s)
    round = 3
    state_target = state_target_1
    state_moving = get_linear_traj(state_now, state_target_1, steps_tot)
    is_target_moving = True
    state_moving = generate_target_trajectory(steps_tot, MPCC.T_s, speed=0.8)
    for i in range(steps_tot): 
        # Solve
        if is_target_moving:
            u_optimal = solver.solve(state_now, state_moving[i,:])
        else:
            u_optimal = solver.solve(state_now, state_target)
        
        # Plant Step
        state_now = robot_plant_step(state_now, u_optimal, MPCC.T_s)

        traj_x.append(state_now[0])
        traj_y.append(state_now[1])
        traj_z.append(state_now[2])
        thrust_history.append(u_optimal)
        
        dist = np.linalg.norm(state_now[:3] - state_moving[i,:3])
        
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
    if is_target_moving:
        plot_TT_3d(state_moving[:,0], state_moving[:,1], state_moving[:,2],
                   np.array(traj_x), np.array(traj_y), np.array(traj_z),
                   thrust_history, MPCC.T_s)
    else:   
        plot_double_target_3d(np.array(traj_x), np.array(traj_y), np.array(traj_z), state_target_1, state_target_2, thrust_history)



if __name__ == "__main__":
    simulation()