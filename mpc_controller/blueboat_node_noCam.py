import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
bluerov_path = os.path.join(current_dir, 'BlueROV2')

if bluerov_path not in sys.path:
    sys.path.append(bluerov_path)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import matplotlib.pyplot as plt

# ROS Msgs
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64  

# NMPC
import numpy as np
from scipy.spatial.transform import Rotation as R  # Updated to avoid deprecation warning

from BlueROV2.config.nmpc_params import BlueBoat_Params
from BlueROV2.nmpc_solver_acados import Acados_Solver_Wrapper
from BlueROV2.core.model import export_vehicle_model
from BlueROV2.estimators.target_estimator import VisualTarget
from BlueROV2.estimators.aekfd import AEKFD

from BlueROV2.utils.plotters import LOS_interactive_viewer, LOS_plot_camera_fov, LOS_plot_dynamics, plot_TT_3d, LOS_plot_dynamics_desired
from BlueROV2.utils.plant_sim import Vehicle_Sim_Utils as Vehicle_Utils 
from BlueROV2.guidance import get_shadow_LOS

class BlueBoatSimMPC(Node):
    def __init__(self):
        super().__init__('blueboat_sim_nmpc_node')

        self.my_params = BlueBoat_Params()
        self.get_logger().info("Initializing Acados solver...")
        self.solver = Acados_Solver_Wrapper(self.my_params)
        self.get_logger().info("Solver Ready!")
        
        acados_model = export_vehicle_model(self.my_params)
        self.ekf = AEKFD(acados_model, self.my_params)

        self.u_previous = np.zeros(self.my_params.nu)
        self.state_now = np.zeros(self.my_params.nx)
        self.ekf.set_state_estimate(self.state_now)
        
        self.ref_target = np.zeros(12)

        self.steps_tot = 2699
        self.actual_step = 0
        self.sim = Vehicle_Utils(self.my_params)

        # Generate imaginary simulated target trajectory
        # self.state_moving = self.sim.get_linear_traj(self.steps_tot, self.my_params.T_s, speed=0.5)
        self.state_moving = self.sim.get_mixed_traj(self.steps_tot, self.my_params.T_s, speed=0.45, is_boat=True)

        self.camera_data = VisualTarget(start_state=self.state_moving[0,:], fov_h=self.my_params.fov_h, fov_v=self.my_params.fov_v, max_dist=20)
        self.seen_it_once = False
        self.wait_here = np.copy(self.state_now[0:12])
        self.camera_noise = np.random.normal(0, 0.000, 3)

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/model/blueboat/odometry', # <--- Listening directly to Gazebo truth
            self.odom_callback,
            qos_sensor
        )

        self.left_pub = self.create_publisher(
            Float64,
             '/model/blueboat/joint/thruster_left_joint/cmd_thrust',
              10 )
        
        self.right_pub = self.create_publisher(
            Float64,
             '/model/blueboat/joint/thruster_right_joint/cmd_thrust',
              10 )
        
        self.get_logger().info('BlueBoat Simulated Control Mode ACTIVE')

        # Data collection for plotting
        self.history_rov_x, self.history_rov_y, self.history_rov_z = [], [], []
        self.history_rov_ph, self.history_rov_th, self.history_rov_ps = [], [], []
        self.history_target = []
        self.thrust_history = []
        
        self.history_est_target_x, self.history_est_target_y, self.history_est_target_z = [], [], []
        self.ref_x, self.ref_y, self.ref_z = [], [], []

    def move(self, sx_speed, dx_speed):
        """Sends commands to left (sx) and right (dx) thruster"""
        msg_sx = Float64()
        msg_dx = Float64()

        msg_sx.data = float(sx_speed)
        msg_dx.data = float(dx_speed)

        self.left_pub.publish(msg_sx)
        self.right_pub.publish(msg_dx)

    def get_state_now(self, msg):
        y = msg.pose.pose.position.y
        x = msg.pose.pose.position.x
        z = msg.pose.pose.position.z

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        r = R.from_quat([qx, qy, qz, qw])
        euler = r.as_euler('xyz', degrees=False)
        
        phi = euler[0]
        theta = euler[1] 
        psi = euler[2]  
        
        u = msg.twist.twist.linear.x
        v = msg.twist.twist.linear.y
        w = msg.twist.twist.linear.z  
        p = msg.twist.twist.angular.x
        q = msg.twist.twist.angular.y
        r_ang = msg.twist.twist.angular.z

        self.state_now = np.array([x, y, z, phi, theta, psi, u, v, w, p, q, r_ang])
        return self.state_now

    def odom_callback(self, msg):
        """Convert ROS ENU Odometry to NMPC NED State and compute control"""
        ros_state = self.get_state_now(msg)
        
        # Logging history
        self.history_rov_x.append(ros_state[0])
        self.history_rov_y.append(ros_state[1])
        self.history_rov_z.append(ros_state[2])
        self.history_rov_ph.append(ros_state[3])
        self.history_rov_th.append(ros_state[4])
        self.history_rov_ps.append(ros_state[5])
        self.history_target.append(self.state_moving[self.actual_step, 0:6])

        # State Estimation
        self.ekf.predict(self.u_previous)
        self.ekf.measurement_update(self.state_now)
        x_est = self.ekf.get_state_estimate()
        d_est = self.ekf.get_disturbance_estimate()
        print(f"Estimated Disturbance: {d_est}")

        # Update simulated target position
        self.camera_data.truth_update(self.state_moving[self.actual_step, 0:6])
        if self.actual_step < (self.steps_tot-1):
            self.actual_step += 1
        else:
            self.get_logger().info("2200 Steps completed! Generating plots...")
            raise SystemExit # This gracefully stops the rclpy.spin() loop
            
        is_visible = self.camera_data.check_visibility(x_est[0:12], self.seen_it_once)
        
        est_target_pos = np.zeros(3)

        # Target Tracking Logic
        if is_visible:
            self.seen_it_once = True
            est_target = self.camera_data.get_camera_estimate(x_est[0:12], dt=self.my_params.T_s, camera_noise=self.camera_noise)
            est_target_pos = est_target[0:3]
            est_target_vel = est_target[3:6]
            
            self.ref_target = get_shadow_LOS(x_est[0:12], est_target_pos, est_target_vel, desired_dist=2.5)
            
        elif not is_visible and self.seen_it_once:
            est_target = self.camera_data.get_camera_estimate(x_est[0:12], dt=self.my_params.T_s, camera_noise=self.camera_noise)
            est_target_pos = est_target[0:3]
            est_target_vel = est_target[3:6]
            
            self.ref_target = get_shadow_LOS(x_est[0:12], est_target_pos, est_target_vel, desired_dist=1.0)

            if self.camera_data.last_seen_t > 30.0:
                self.ref_target = np.copy(x_est[0:12])
                self.ref_target[6:12] = 0.0
                self.ref_target[4] = 0 
                self.seen_it_once = False
                self.wait_here = np.copy(x_est[0:12])
        else:
            self.ref_target = np.copy(self.wait_here[0:12])
            est_target_pos = self.wait_here[0:3]

        # Log estimation and references
        self.history_est_target_x.append(est_target_pos[0])
        self.history_est_target_y.append(est_target_pos[1])
        self.history_est_target_z.append(est_target_pos[2])

        ref_safe = np.atleast_2d(self.ref_target)  
        self.ref_x.append(ref_safe[0, 0])
        self.ref_y.append(ref_safe[0, 1])
        self.ref_z.append(ref_safe[0, 2])

        # Control
        u_optimal = self.solver.solve(x_est, self.ref_target, disturbance=d_est)
        self.move(u_optimal[0], u_optimal[1])
        
        self.u_previous = u_optimal
        self.thrust_history.append(u_optimal)

def main(args=None):
    rclpy.init(args=args)
    node = BlueBoatSimMPC()
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # Safely attempt to zero the thrusters if ROS is still alive
        try:
            if rclpy.ok():
                node.move(0.0, 0.0)
        except Exception:
            pass # Context invalidated by Ctrl+C, move on to plotting
        
        # Convert lists to arrays
        rov_x = np.array(node.history_rov_x)
        rov_y = np.array(node.history_rov_y)
        rov_z = np.array(node.history_rov_z)
        rov_ph = np.array(node.history_rov_ph)
        rov_th = np.array(node.history_rov_th)
        rov_ps = np.array(node.history_rov_ps)
        
        target_data = np.array(node.history_target)
        est_target_data_x = np.array(node.history_est_target_x)
        est_target_data_y = np.array(node.history_est_target_y)
        est_target_data_z = np.array(node.history_est_target_z)
        
        ref_data_x = np.array(node.ref_x)
        ref_data_y = np.array(node.ref_y)
        ref_data_z = np.array(node.ref_z)
        thrust_hist = np.array(node.thrust_history)

        dt = node.my_params.T_s

        # Find the shortest array length to prevent broadcast errors from ^C
        min_len = min(
            len(rov_x), len(ref_data_x), len(est_target_data_x), 
            len(thrust_hist), len(target_data)
        )

        if min_len > 0:
            # Slice all arrays perfectly to min_len
            rov_x, rov_y, rov_z = rov_x[:min_len], rov_y[:min_len], rov_z[:min_len]
            rov_ph, rov_th, rov_ps = rov_ph[:min_len], rov_th[:min_len], rov_ps[:min_len]
            ref_data_x, ref_data_y, ref_data_z = ref_data_x[:min_len], ref_data_y[:min_len], ref_data_z[:min_len]
            est_target_data_x, est_target_data_y, est_target_data_z = est_target_data_x[:min_len], est_target_data_y[:min_len], est_target_data_z[:min_len]
            target_data = target_data[:min_len, :]
            thrust_hist = thrust_hist[:min_len]

            # Save Data
            save_path = os.path.join(current_dir, 'BB_sim_test.npz')
            np.savez(save_path, 
                            rov_x=rov_x, rov_y=rov_y, rov_z=rov_z, 
                            rov_ph=rov_ph, rov_th=rov_th, rov_ps=rov_ps, 
                            target=target_data, ref_x=ref_data_x, 
                            est_target_data_x=est_target_data_x, est_target_data_y=est_target_data_y, est_target_data_z=est_target_data_z,
                            ref_y=ref_data_y, ref_z=ref_data_z, 
                            thrust=thrust_hist, dt=dt)

            print("Plotting results...")
            
            # 1. Main 3D plot (Target Truth vs ROV)
            plot_TT_3d(target_data[:,0], target_data[:,1], target_data[:,2],
                ref_data_x, ref_data_y, ref_data_z, 
                rov_x, rov_y, rov_z,
                rov_ph, rov_th, rov_ps, 
                thrust_hist, dt
            )

            # 2. Dynamics desired (Error tracking)
            LOS_plot_dynamics_desired(rov_x, rov_y, rov_z, target_data, dt, desired_dist=2.5, ref_x=ref_data_x, ref_y=ref_data_y, ref_z=ref_data_z)

            # 3. Target estimation 3D Plot (EKF Estimate vs ROV)
            plot_TT_3d(est_target_data_x, est_target_data_y, est_target_data_z,
                ref_data_x, ref_data_y, ref_data_z, 
                rov_x, rov_y, rov_z,
                rov_ph, rov_th, rov_ps, 
                thrust_hist, dt
            )

            # 4. Camera FOV Plot
            LOS_plot_camera_fov(rov_x, rov_y, rov_z, rov_ps, rov_th, target_data, dt)
            
            # Print Error stats
            try:
                node.sim.get_error_avg_std([rov_x, rov_y, rov_z], target_data[:,:3].T, [ref_data_x, ref_data_y, ref_data_z])
            except Exception as e:
                pass

            plt.show()
            
        node.destroy_node()
        
        # Check if rclpy is still valid before attempting shutdown
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()