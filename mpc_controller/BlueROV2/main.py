import numpy as np
import matplotlib.pyplot as plt
import time
from estimators.aekfd import AEKFD as EKF
from estimators.target_estimator import VisualTarget

from nmpc_solver_acados import Acados_Solver_Wrapper
from config.nmpc_params import BlueROV_Params, BlueBoat_Params
from core.model import export_vehicle_model
from utils.plotters import LOS_plot_dynamics, plot_TT_3d, LOS_plot_camera_fov
from utils.plant_sim import Vehicle_Sim_Utils as Vehicle_Utils 
from guidance import get_shadow_ref, get_shadow_traj
def simulation():
    vehicle_type = "ROV"
    state_now = np.zeros(12)
    
    if vehicle_type == "ROV":
        my_params = BlueROV_Params()
        print("Loading BlueROV2 Heavy")
        state_now[2] = -2
    elif vehicle_type == "BOAT":
        my_params = BlueBoat_Params()
    sim = Vehicle_Utils(my_params)

    print("Compiling Acados Solver...")
    solver = Acados_Solver_Wrapper(my_params)
    print("Compilation Complete.")



    # I'll clean this up i swear
    traj_x, traj_y, traj_z = [], [], []
    traj_phi, traj_theta, traj_psi = [], [], []

    EKFtraj_x, EKFtraj_y, EKFtraj_z = [], [], []
    EKFtraj_phi, EKFtraj_theta, EKFtraj_psi = [], [], []

    ref_x, ref_y, ref_z = [], [], []

    target_estimation_x, target_estimation_y, target_estimation_z = [], [], []
    est_target_pos = np.zeros(3)

    thrust_history = [] 
    u_previous = np.zeros(my_params.nu)
    MAX_DELTA = my_params.THRUST_MAX * my_params.T_s *1000

    t_simulation = 30 #sec
    steps_tot = int(t_simulation /  my_params.T_s)

    # Assuming that this is the data we get from our ideal camera system
    state_moving = sim.generate_target_trajectory(steps_tot, my_params.T_s, speed=0.9)
    t0 = time.time()


    # TO MODIFY: JUST KEEPING TRACK OF WHAT TYPE OF MISSION WE ARE DOING
    round = 4
    if round == 3:      # Straight line
        state_moving = sim.get_linear_traj(steps_tot, my_params.T_s, speed=0.9)
    elif round == 4:    # Mad fella trajectory
        state_moving = sim.generate_target_trajectory(steps_tot, my_params.T_s, speed=0.9)


    #EKF Setup
    acados_model = export_vehicle_model(my_params)
    ekf = EKF(acados_model, my_params)
    ekf.set_state_estimate(state_now)

    # State estimate initialization for EKF with disturbance states
    state_est = np.zeros(12)
    state_est[0:12] = np.copy(state_now)

    # Too many flags
    testing_EKF = True
    is_there_noise = True
    noise_ekf = my_params.noise_ekf if is_there_noise else np.array([0.0]*12)    
    print("noise_ekf:", noise_ekf)
    # Camera Model Setup
    camera_data = VisualTarget(start_state=state_moving[0,:], fov_h=my_params.fov_h, fov_v=my_params.fov_v, max_dist=10)
    camera_noise = np.random.normal(0, 0.006, 3) if is_there_noise else np.array([0.0]*3)
    seen_it_once = False
    wait_here = np.copy(state_now[0:12])

    # External Disturbance setup:
    is_there_disturbance = False
    estimated_disturbance = np.zeros(6)
    force_world = np.array([-30,-30, 0])
    tether_disturbance = np.array([0,0,0,0,0,0])
    real_disturbance = np.zeros(6)

    if round > 2:
        print("BLEEEP")
        for i in range(steps_tot): 
            camera_data.truth_update(state_moving[i,0:6])

            is_visible = camera_data.check_visibility(state_est[0:12], seen_it_once)
            if is_visible:
                # If visible modify the seen flag, estimate target position and get the trajectory
                seen_it_once = True
                est_target = camera_data.get_camera_estimate(state_est[0:12], dt = my_params.T_s, camera_noise = camera_noise)
                est_target_pos = est_target[0:3]
                est_target_vel = est_target[3:6]

                #ref_guidance = get_shadow_traj(state_est[0:12], est_target_pos, est_target_vel, dt = my_params.T_s,horizon_N = my_params.N+1, desired_dist=2.5)
                ref_guidance = get_shadow_ref(state_est[0:12], est_target_pos, est_target_vel, desired_dist=2.5)

            elif not is_visible and seen_it_once:

                # Visible before but not now: use last seen position and et imaginary reference based on last seen position
                est_target = camera_data.get_camera_estimate(state_est[0:12], dt = my_params.T_s, camera_noise = camera_noise)
                est_target_pos = est_target[0:3]
                est_target_vel = est_target[3:6]
                ref_guidance = get_shadow_ref(state_est[0:12], est_target_pos, est_target_vel, desired_dist=2.0)
                
                if camera_data.last_seen_t > 5.0:
                    # If not seen for more than 5 seconds, just stay still
                    ref_guidance = state_est[0:12]
                    ref_guidance[6:12] = 0.0
                    ref_guidance[4] = 0 
                    seen_it_once = False
                    wait_here = state_est[0:12]
            else:
                # Not visible and never seen: stay still till i see something
                ref_guidance = wait_here[0:12]

            u_optimal = solver.solve(state_est, ref_guidance,  disturbance=estimated_disturbance)
            u_optimal = cap_input(u_previous, u_optimal, MAX_DELTA)
            u_previous = u_optimal

            print("u_optimal: ", u_optimal)

            # Plant Step [Imagine this as GPS data cause we dont have GPS data AND RK4might be a little different from drone state estimation so there is error (good i guess otherwise we are just simulating in same perfect hysics)]
            if is_there_disturbance == False:
                real_disturbance[0:6] = np.zeros(6)
            elif i > 200:
                real_disturbance[0:3] = sim.force_w2b(state_est, force_world*np.abs(np.sin(i/400)))
            else:
                real_disturbance[0:3] = sim.force_w2b(state_est, force_world*0)
            state_now = sim.robot_plant_step_RK4(state_now, u_optimal, my_params.T_s, disturbance = real_disturbance)
            

            if testing_EKF:
                if is_there_noise:
                    noise_ekf = my_params.noise_ekf
                else:
                    noise_ekf = np.random.normal(0, [0.0]*12)
                measured_state = state_now[0:12] + noise_ekf

                ekf.predict(u_optimal)
                ekf.measurement_update(measured_state)
                state_est = ekf.get_state_estimate()
                estimated_disturbance = ekf.get_disturbance_estimate()
            else:
                state_est = np.copy(state_now[0:12])
            


            # Keep track of data for plotting... i'll have to clean this
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

            ref_safe = np.atleast_2d(ref_guidance)  
            ref_x.append(ref_safe[0, 0])
            ref_y.append(ref_safe[0, 1])
            ref_z.append(ref_safe[0, 2])
            if is_visible:
                target_estimation_x.append(est_target_pos[0])
                target_estimation_y.append(est_target_pos[1])
                target_estimation_z.append(est_target_pos[2])

            camera_noise = np.random.normal(0, 0.006, 3)

    # Straight line trajectory
    elif round == 3:
        for i in range(steps_tot): 
            
            u_optimal = solver.solve(state_now, state_moving[i,:])
            u_optimal = cap_input(u_previous, u_optimal, MAX_DELTA)
            u_previous = u_optimal
            
            state_now = sim.robot_plant_step(state_now, u_optimal, my_params.T_s, real_disturbance)

            traj_x.append(state_now[0])
            traj_y.append(state_now[1])
            traj_z.append(state_now[2])
            traj_phi.append(state_now[3])   # Roll
            traj_theta.append(state_now[4]) # Pitch
            traj_psi.append(state_now[5])   # Yaw
            thrust_history.append(u_optimal)
    



    # Plotting
    t_end = time.time()
    t_sum = t_end - t0
    print(f"{t_sum:.4f}s is the total computational time!")
    
    print("Plotting results...")
    thrust_history = np.array(thrust_history)
    if round >= 3:
        
        plot_TT_3d(state_moving[:,0], state_moving[:,1], state_moving[:,2],
                    ref_x, ref_y, ref_z, # Reference
                    np.array(traj_x), np.array(traj_y), np.array(traj_z),    # ROV Position
                    np.array(traj_phi), np.array(traj_theta), np.array(traj_psi), # ROV Angles
                    thrust_history, my_params.T_s
                )
        
        LOS_plot_dynamics(traj_x, traj_y, traj_z, state_moving, my_params.T_s, desired_dist=2.0)

        if testing_EKF:
            plot_TT_3d(np.array(target_estimation_x), np.array(target_estimation_y), np.array(target_estimation_z),
                    ref_x, ref_y, ref_z, # Reference
                    np.array(traj_x), np.array(traj_y), np.array(traj_z),    # ROV Position
                    np.array(traj_phi), np.array(traj_theta), np.array(traj_psi), # ROV Angles
                    thrust_history, my_params.T_s
                )
            LOS_plot_dynamics(EKFtraj_x, EKFtraj_y, EKFtraj_z, state_moving, my_params.T_s, desired_dist=2.0)
            
        LOS_plot_camera_fov(traj_x, traj_y, traj_z, traj_psi, traj_theta, state_moving, my_params.T_s)
        sim.get_error_avg_std([traj_x, traj_y, traj_z], state_moving[:,:3].T, [ref_x, ref_y, ref_z])
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