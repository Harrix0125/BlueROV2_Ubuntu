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
from geometry_msgs.msg import Point

# NMPC
import numpy as np
import scipy.spatial.transform.rotation as R

from BlueROV2.config.nmpc_params import BlueBoat_Params
from BlueROV2.nmpc_solver_acados import Acados_Solver_Wrapper
from BlueROV2.core.model import export_vehicle_model
from BlueROV2.estimators.target_estimator import VisualTarget
from BlueROV2.estimators.aekfd import AEKFD

from BlueROV2.utils.plotters import LOS_interactive_viewer, LOS_plot_camera_fov, LOS_plot_dynamics, plot_TT_3d
from BlueROV2.utils.plant_sim import Vehicle_Sim_Utils as Vehicle_Utils 
from BlueROV2.guidance import get_shadow_ref, get_shadow_traj, get_shadow_LOS

class BlueBoatMPC(Node):
    def __init__(self):
        super().__init__('blueboat_nmpc_node')

        self.my_params = BlueBoat_Params()
        self.get_logger().info("Initilizing Acados solver...")
        self.solver = Acados_Solver_Wrapper(self.my_params)
        self.get_logger().info("Solver Ready!")
        acados_model = export_vehicle_model(self.my_params)
        self.ekf = AEKFD(acados_model, self.my_params)

        self.u_previous = np.zeros(self.my_params.nu)
        self.state_now = np.zeros(self.my_params.nx)

        self.ekf.set_state_estimate(self.state_now)
        self.ref_target = np.zeros(12)

        self.steps_tot = 1000
        self.actual_step = 0
        self.sim = Vehicle_Utils(self.my_params)

        self.state_moving = self.sim.get_linear_traj(self.steps_tot, self.my_params.T_s, speed=0.75)

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
            '/model/blueboat/odometry', 
            self.odom_callback,
            qos_sensor
        )

        self.left_pub = self.create_publisher(Float64, '/model/blueboat/joint/thruster_left_joint/cmd_thrust', 10)
        self.right_pub = self.create_publisher(Float64, '/model/blueboat/joint/thruster_right_joint/cmd_thrust', 10)
        
        # Camera Node
        self.use_real_camera = True 
        self.real_target_relative = np.zeros(3)
        self.last_camera_update = 0.0
        
        if self.use_real_camera:
            self.vision_sub = self.create_subscription(
                Point,
                '/sensor/apriltag/relative_pos',
                self.vision_callback,
                10
            )
            self.get_logger().info('Real Camera Mode ACTIVE')
        else:
            self.get_logger().info('Simulated Camera Mode ACTIVE')

        self.get_logger().info('BlueBoat is ready ')

        # History arrays
        self.history_rov_x = []
        self.history_rov_y = []
        self.history_rov_z = []
        self.history_rov_ph = []
        self.history_rov_th = []
        self.history_rov_ps = []
        self.history_target = []
        self.thrust_history = []
        
        self.history_real_target_x = []
        self.history_real_target_y = []
        self.history_real_target_z = []

        self.ref_x = []
        self.ref_y = []
        self.ref_z = []

    def move(self, sx_speed, dx_speed):
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
        
        r = R.Rotation.from_quat([qx, qy, qz, qw])
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
    
    def vision_callback(self, msg):
        self.real_target_relative = np.array([msg.x, msg.y, msg.z])
        self.last_camera_update = time.time()
        self.seen_it_once = True

    def update_target_simulated(self, x_est):
        self.camera_data.truth_update(self.state_moving[self.actual_step, 0:6])
        
        if self.actual_step < (self.steps_tot-1):
            self.actual_step += 1
            
        is_visible = self.camera_data.check_visibility(self.state_now, self.seen_it_once)
        est_target_pos = np.zeros(3) # Default value to prevent append errors
        
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
                self.ref_target = x_est[0:12]
                self.ref_target[6:12] = 0.0
                self.ref_target[4] = 0 
                self.seen_it_once = False
                self.wait_here = x_est[0:12]
        else:
            self.ref_target = self.wait_here[0:12]
            
        # ALWAYS append to keep array sizes matching the odometry arrays
        self.history_real_target_x.append(est_target_pos[0])
        self.history_real_target_y.append(est_target_pos[1])
        self.history_real_target_z.append(est_target_pos[2])

    def update_target_real(self, x_est):
        is_actually_visible = (time.time() - self.last_camera_update) < 1.0

        est_target_pos = np.zeros(3)
        est_target_vel = np.zeros(3)

        if is_actually_visible:
            self.seen_it_once = True
            boat_pos = x_est[0:3]
            boat_yaw = x_est[5]
            
            cam_dx = self.real_target_relative[0]
            cam_dist = self.real_target_relative[2]
            
            target_world_x = boat_pos[0] + (cam_dist * np.cos(boat_yaw)) - (cam_dx * np.sin(boat_yaw))
            target_world_y = boat_pos[1] + (cam_dist * np.sin(boat_yaw)) + (cam_dx * np.cos(boat_yaw))
            target_world_z = boat_pos[2] 
            
            est_target_pos = np.array([target_world_x, target_world_y, target_world_z])
            self.ref_target = get_shadow_LOS(x_est[0:12], est_target_pos, est_target_vel, desired_dist=2.0)

        elif not is_actually_visible and self.seen_it_once:
            self.ref_target = get_shadow_LOS(x_est[0:12], est_target_pos, est_target_vel, desired_dist=1.0)
            
            if (time.time() - self.last_camera_update) > 5.0:
                self.ref_target = x_est[0:12]
                self.ref_target[6:12] = 0.0
                self.seen_it_once = False
                self.wait_here = x_est[0:12]
        else:
             self.ref_target = self.wait_here[0:12]
             
        # ALWAYS append to keep array sizes matching the odometry arrays
        self.history_real_target_x.append(est_target_pos[0])
        self.history_real_target_y.append(est_target_pos[1])
        self.history_real_target_z.append(est_target_pos[2])


    def odom_callback(self, msg):
        ros_state = self.get_state_now(msg)
        
        self.history_rov_x.append(ros_state[0])
        self.history_rov_y.append(ros_state[1])
        self.history_rov_z.append(ros_state[2])
        self.history_rov_ph.append(ros_state[3])
        self.history_rov_th.append(ros_state[4])
        self.history_rov_ps.append(ros_state[5])
        self.history_target.append(self.state_moving[self.actual_step, 0:6])

        self.ekf.predict(self.u_previous)
        self.ekf.measurement_update(self.state_now)

        x_est = self.ekf.get_state_estimate()
        d_est = self.ekf.get_disturbance_estimate()
        print("position : ", x_est[0:6])

        if self.use_real_camera:
            self.update_target_real(x_est)
        else:
            self.update_target_simulated(x_est)

        ref_safe = np.atleast_2d(self.ref_target)  
        self.ref_x.append(ref_safe[0, 0])
        self.ref_y.append(ref_safe[0, 1])
        self.ref_z.append(ref_safe[0, 2])

        u_optimal = self.solver.solve(x_est, self.ref_target, disturbance = d_est)
        
        self.move(u_optimal[0], u_optimal[1])
        self.u_previous = u_optimal
        self.thrust_history.append(u_optimal)


def main(args=None):
    rclpy.init(args=args)
    node = BlueBoatMPC()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        zero_msg = Float64()
        zero_msg.data = 0.0
        node.left_pub.publish(zero_msg)
        node.right_pub.publish(zero_msg)

        # 1. Convert everything to numpy arrays
        rov_x = np.array(node.history_rov_x)
        rov_y = np.array(node.history_rov_y)
        rov_z = np.array(node.history_rov_z)
        rov_ph = np.array(node.history_rov_ph)
        rov_th = np.array(node.history_rov_th)
        rov_ps = np.array(node.history_rov_ps)
        
        target_data = np.array(node.history_target)
        real_target_data_x = np.array(node.history_real_target_x)
        real_target_data_y = np.array(node.history_real_target_y)
        real_target_data_z = np.array(node.history_real_target_z)
        
        ref_data_x = np.array(node.ref_x)
        ref_data_y = np.array(node.ref_y)
        ref_data_z = np.array(node.ref_z)
        thrust_history = np.array(node.thrust_history)

        # 2. Find the minimum length among all dynamic arrays to prevent broadcasting errors
        lengths = [len(rov_x), len(ref_data_x), len(thrust_history), len(node.state_moving), len(real_target_data_x)]
        if min(lengths) == 0:
            print("No data recorded. Shutting down safely.")
            node.destroy_node()
            rclpy.shutdown()
            return
            
        min_len = min(lengths)

        # 3. Truncate ALL arrays identically so they share the exact same shape
        rov_x = rov_x[:min_len]
        rov_y = rov_y[:min_len]
        rov_z = rov_z[:min_len]
        rov_ph = rov_ph[:min_len]
        rov_th = rov_th[:min_len]
        rov_ps = rov_ps[:min_len]
        
        target_data = target_data[:min_len, :]
        state_moving_sliced = node.state_moving[:min_len, :]
        
        real_target_data_x = real_target_data_x[:min_len]
        real_target_data_y = real_target_data_y[:min_len]
        real_target_data_z = real_target_data_z[:min_len]
        
        ref_data_x = ref_data_x[:min_len]
        ref_data_y = ref_data_y[:min_len]
        ref_data_z = ref_data_z[:min_len]
        thrust_history = thrust_history[:min_len]
        dt = node.my_params.T_s

        # 4. Save data safely
        save_path = os.path.join(current_dir, 'BB_home_test.npz')
        np.savez(save_path, 
                 rov_x=rov_x, rov_y=rov_y, rov_z=rov_z, 
                 rov_ph=rov_ph, rov_th=rov_th, rov_ps=rov_ps, 
                 target=target_data, 
                 ref_x=ref_data_x, ref_y=ref_data_y, ref_z=ref_data_z, 
                 real_target_data_x=real_target_data_x, 
                 real_target_data_y=real_target_data_y, 
                 real_target_data_z=real_target_data_z,
                 thrust=thrust_history, dt=dt)

        # 5. Plot the aligned data
        plot_TT_3d(state_moving_sliced[:,0], state_moving_sliced[:,1], state_moving_sliced[:,2],
                   ref_data_x, ref_data_y, ref_data_z, 
                   rov_x, rov_y, rov_z,    
                   rov_ph, rov_th, rov_ps, 
                   thrust_history, dt)

        LOS_plot_camera_fov(rov_x, rov_y, rov_z, rov_ps, rov_th, state_moving_sliced, dt)

        if len(rov_x) > 0:
            plt.show()
            
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()