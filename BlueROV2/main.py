import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

from nmpc_solver_acados import Acados_Solver_Wrapper
from nmpc_params import NMPC_params as MPCC
from utils import get_linear_traj, robot_plant_step, generate_target_trajectory, get_standoff_reference
from plotters import plot_double_target_3d, plot_TT_3d, LOS_plot_heading_error, LOS_plot_standoff_tracking
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
    traj_phi, traj_theta, traj_psi = [], [], []
    thrust_history = []   
    t_simulation = 100 #sec
    steps_tot = int(t_simulation /  MPCC.T_s)

    # Assuming that this is the data we get from our ideal camera system
    state_target = state_target_1
    state_moving = generate_target_trajectory(steps_tot, MPCC.T_s, speed=1)
    t0 = time.time()

    # TO MODIFY: JUST KEEPING TRACK OF WHAT TYPE OF MISSION WE ARE DOING
    round = 4
    

    if round ==4:
        for i in range(steps_tot): 
            # Solve
            current_target = state_moving[i,:]
            ref_guidance = get_standoff_reference(state_now, current_target, desired_dist=2.0, lookahead=1.0)
            u_optimal = solver.solve(state_now, ref_guidance)
            # Plant Step
            state_now = robot_plant_step(state_now, u_optimal, MPCC.T_s)

            # Keep track of data for plotting
            traj_x.append(state_now[0])
            traj_y.append(state_now[1])
            traj_z.append(state_now[2])
            traj_phi.append(state_now[3])   # Roll
            traj_theta.append(state_now[4]) # Pitch
            traj_psi.append(state_now[5])   # Yaw
            thrust_history.append(u_optimal)

    elif round == 3:
        for i in range(steps_tot): 
            # Solve
            u_optimal = solver.solve(state_now, state_moving[i,:])
            
            # Plant Step
            state_now = robot_plant_step(state_now, u_optimal, MPCC.T_s)

            traj_x.append(state_now[0])
            traj_y.append(state_now[1])
            traj_z.append(state_now[2])
            traj_phi.append(state_now[3])   # Roll
            traj_theta.append(state_now[4]) # Pitch
            traj_psi.append(state_now[5])   # Yaw
            thrust_history.append(u_optimal)
    elif round < 3:
        for i in range(steps_tot): 
            # Solve
            u_optimal = solver.solve(state_now, state_target)
            
            # Plant Step
            state_now = robot_plant_step(state_now, u_optimal, MPCC.T_s)

            traj_x.append(state_now[0])
            traj_y.append(state_now[1])
            traj_z.append(state_now[2])
            traj_phi.append(state_now[3])   # Roll
            traj_theta.append(state_now[4]) # Pitch
            traj_psi.append(state_now[5])   # Yaw
            thrust_history.append(u_optimal)
            
            dist = np.linalg.norm(state_now[:3] - state_target[:3])
            
            if dist < 0.05 and round == 2:
                print(f"Target Reached at step {i}!")
                break
            elif dist < 0.05 and round == 1:
                print(f"First Target Reached at step {i}, setting new target.")
                state_target = state_target_2
                round += 1


    # 3. Plotting
    t_end = time.time()
    t_sum = t_end - t0
    print(f"{t_sum:.4f}s is the total computational time!")
    
    print("Plotting results...")
    thrust_history = np.array(thrust_history)
    if round == 3:
        plot_TT_3d(
                    state_moving[:,0], state_moving[:,1], state_moving[:,2], # Reference
                    np.array(traj_x), np.array(traj_y), np.array(traj_z),    # ROV Position
                    np.array(traj_phi), np.array(traj_theta), np.array(traj_psi), # ROV Angles
                    thrust_history, MPCC.T_s
                )
    elif round <3:   
        plot_double_target_3d(np.array(traj_x), np.array(traj_y), np.array(traj_z), state_target_1, state_target_2, thrust_history)
    elif round ==4:
        LOS_plot_standoff_tracking(
                    traj_x, 
                    traj_y, 
                    traj_z, 
                    state_moving,   # This is your target array (steps, 12)
                    MPCC.T_s,       # Time step
                    desired_dist=2.0
                )
        LOS_plot_heading_error(
            traj_x, 
            traj_y, 
            traj_psi,        # Pass the Yaw/Psi history
            state_moving,    # Target data
            MPCC.T_s
        )



if __name__ == "__main__":
    simulation()