import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from AEKFD import AEKFD as EKF
from target_estimator import VisualTarget
import casadi as cas

from nmpc_solver_acados import Acados_Solver_Wrapper
from nmpc_params import NMPC_params as MPCC
from utils import utils
from plotters import LOS_plot_dynamics, plot_double_target_3d, plot_TT_3d, LOS_plot_camera_fov
def simulation():
    # Initialize the new Acados solver
    # This will trigger code generation and compilation (takes time once)
    print("Compiling Acados Solver...")
    solver = Acados_Solver_Wrapper()
    print("Compilation Complete.")

    state_now = np.zeros(12)
    state_now[2] = -2
    #state_now[6] = -1 / (MPCC.R_THRUST * 1200)

    estimated_disturbance = np.zeros(6)

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

    EKFtraj_x, EKFtraj_y, EKFtraj_z = [], [], []
    EKFtraj_phi, EKFtraj_theta, EKFtraj_psi = [], [], []

    ref_x, ref_y, ref_z = [], [], []

    thrust_history = [] 
    u_previous = np.zeros(8)
    MAX_DELTA = MPCC.THRUST_MAX * MPCC.T_s * 2

    t_simulation = 30 #sec
    steps_tot = int(t_simulation /  MPCC.T_s)

    # Assuming that this is the data we get from our ideal camera system
    state_target = state_target_1
    state_moving = utils.generate_target_trajectory(steps_tot, MPCC.T_s, speed=0.9)
    t0 = time.time()

    # TO MODIFY: JUST KEEPING TRACK OF WHAT TYPE OF MISSION WE ARE DOING
    round = 4
    if round == 3:
        state_moving = utils.get_linear_traj(steps_tot, MPCC.T_s, speed=0.9)
    elif round == 4:
        state_moving = utils.generate_target_trajectory(steps_tot, MPCC.T_s, speed=0.9)

    #EKF Setup
    ekf = EKF()
    ekf.set_state_estimate(state_now)

    # State estimate initialization for EKF with disturbance states
    state_est = np.zeros(12)
    state_est[0:12] = np.copy(state_now)

    testing_EKF = True
    is_there_noise = False
    noise_ekf = np.random.normal(0, [0.10, 0.10, 0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) if is_there_noise else np.array([0.0]*12)    
    print("noise_ekf:", noise_ekf)
    # Camera Model Setup
    camera_data = VisualTarget(start_state=state_moving[0,:], fov_h=90, fov_v=80, max_dist=10)
    camera_noise = np.random.normal(0, 0.1, 3) if is_there_noise else np.array([0.0]*3)
    seen_it_once = False

    if round > 2:
        for i in range(steps_tot): 
            camera_data.truth_update(state_moving[i,0:6])

            is_visible = camera_data.check_visibility(state_est[0:12])
            if is_visible:
                seen_it_once = True

                # Update target estimate from camera and get state
                est_target = camera_data.get_camera_estimate(state_est[0:12], dt = MPCC.T_s, camera_noise = camera_noise)
                est_target_pos = est_target[0:3]
                est_target_vel = est_target[3:6]

                # Get guidance reference
                ref_guidance = utils.get_shadow_ref(state_est[0:12], est_target_pos, est_target_vel, desired_dist=2.5)

            elif not is_visible and seen_it_once:

                # Visible before but not now: use last seen position
                est_target = camera_data.get_camera_estimate(state_est[0:12], dt = MPCC.T_s, camera_noise = camera_noise)
                est_target_pos = est_target[0:3]
                est_target_vel = est_target[3:6]

                # Get imaginary reference based on last seen position
                ref_guidance = utils.get_shadow_ref(state_est[0:12], est_target_pos, est_target_vel, desired_dist=4.0)
                if camera_data.last_seen_t > 5.0:
                    # If not seen for more than 5 seconds, just stay still
                    ref_guidance = state_est[0:12]
                    ref_guidance[6:12] = 0.0
                    ref_guidance[4] = 0 
                    seen_it_once = False
            else:
                # Not visible and never seen: stay still till i see something
                ref_guidance = state_est[0:12]
            
            # Solve NMPC
            u_optimal = solver.solve(state_est, ref_guidance,  disturbance=estimated_disturbance)
            u_optimal = cap_input(u_previous, u_optimal, MAX_DELTA)
            u_previous = u_optimal
            # Plant Step [Imagine this as GPS]
            state_now = utils.robot_plant_step_RK4(state_now, u_optimal, MPCC.T_s)
            

            if testing_EKF:
                # EKF:
                if is_there_noise:
                    noise_ekf = np.random.normal(0, [0.10, 0.10, 0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
                else:
                    noise_ekf = np.random.normal(0, [0.0]*12)
                measured_state = state_now[0:12] + noise_ekf
                # Updates EKF P and state estimate
                ekf.predict(u_optimal)
                # Gets new state estimate with K gain
                ekf.measurement_update(measured_state)
                state_est = ekf.get_state_estimate()
                estimated_disturbance = ekf.get_disturbance_estimate()
            else:
                state_est = np.copy(state_now[0:12])
            


            # Keep track of data for plotting
            traj_x.append(state_now[0])
            traj_y.append(state_now[1])
            traj_z.append(state_now[2])
            traj_phi.append(state_now[3])   # Roll
            traj_theta.append(state_now[4]) # Pitch
            traj_psi.append(state_now[5])   # Yaw
            thrust_history.append(u_optimal)
            
            EKFtraj_x.append(state_est[0])
            EKFtraj_y.append(state_est[1])
            EKFtraj_z.append(state_est[2])
            EKFtraj_phi.append(state_est[3])   # Roll
            EKFtraj_theta.append(state_est[4]) # Pitch
            EKFtraj_psi.append(state_est[5])   # Yaw

            ref_x.append(ref_guidance[0])
            ref_y.append(ref_guidance[1])
            ref_z.append(ref_guidance[2])

    elif round == 3:
        for i in range(steps_tot): 
            # Solve
            u_optimal = solver.solve(state_now, state_moving[i,:])
            u_optimal = cap_input(u_previous, u_optimal, MAX_DELTA)
            u_previous = u_optimal
            # Plant Step
            state_now = utils.robot_plant_step(state_now, u_optimal, MPCC.T_s)

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
            u_optimal = cap_input(u_previous, u_optimal, MAX_DELTA)
            u_previous = u_optimal
            # Plant Step
            state_now = utils.robot_plant_step(state_now, u_optimal, MPCC.T_s)

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
    if round <3:   
        plot_double_target_3d(np.array(traj_x), np.array(traj_y), np.array(traj_z), state_target_1, state_target_2, thrust_history)
    elif round >= 3:
        plot_TT_3d(
                    state_moving[:,0], state_moving[:,1], state_moving[:,2], # Reference
                    np.array(traj_x), np.array(traj_y), np.array(traj_z),    # ROV Position
                    np.array(traj_phi), np.array(traj_theta), np.array(traj_psi), # ROV Angles
                    thrust_history, MPCC.T_s
                )
        LOS_plot_dynamics(traj_x, traj_y, traj_z, state_moving, MPCC.T_s, desired_dist=2.0)

        if testing_EKF:
            LOS_plot_dynamics(EKFtraj_x, EKFtraj_y, EKFtraj_z, state_moving, MPCC.T_s, desired_dist=2.0)
            
        LOS_plot_camera_fov(traj_x, traj_y, traj_z, traj_psi, traj_theta, state_moving, MPCC.T_s)
        utils.get_error_avg_std([traj_x, traj_y, traj_z], state_moving[:,:3].T, [ref_x, ref_y, ref_z])
        plt.show()

def cap_input(u_previous, u_optimal, MAX_DELTA = 35.0):
    u_output = np.zeros(len(u_optimal))
    for i in range(len(u_optimal)):
        if u_optimal[i] >= u_previous[i] + MAX_DELTA:
            u_output[i] = u_previous[i] + MAX_DELTA
        elif u_optimal[i] <= u_previous[i] - MAX_DELTA:
            u_output[i] = u_previous[i] - MAX_DELTA
        else:
            u_output[i] = u_optimal[i]
    return u_output
if __name__ == "__main__":
    simulation()