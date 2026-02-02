import sys
import os

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
import scipy.spatial.transform.rotation as R

from BlueROV2.config.nmpc_params import GazeboROV_Params

from BlueROV2.nmpc_solver_acados import Acados_Solver_Wrapper

from BlueROV2.core.model import export_vehicle_model

from BlueROV2.estimators.target_estimator import VisualTarget
from BlueROV2.estimators.aekfd import AEKFD

from BlueROV2.utils.plotters import LOS_interactive_viewer, LOS_plot_camera_fov
from BlueROV2.utils.plant_sim import Vehicle_Sim_Utils as Vehicle_Utils 

from BlueROV2.guidance import get_shadow_ref, get_shadow_traj


class BlueROVMPC(Node):
    def __init__(self):
        super().__init__('bluerov_nmpc_node')

        self.my_params = GazeboROV_Params()
        self.get_logger().info("Initializing Acados Solver...")
        self.solver = Acados_Solver_Wrapper(self.my_params)
        self.get_logger().info("Solver Ready!")
        acados_model = export_vehicle_model(self.my_params)
        self.ekf = AEKFD(acados_model, self.my_params)

        self.u_previous = np.zeros(8)

        self.state_now = np.zeros(12) 
        self.state_now[2] = -2
        self.ekf.set_state_estimate(self.state_now)

        self.ref_target = np.zeros(12)
        # self.ref_target[2] = -1.0  
        # self.ref_target[1] = 5.0  
        # self.ref_target[0] = 0.0 
        # self.ref_target[5] = 1.55


        self.steps_tot = 1000
        self.actual_step = 0
        self.sim = Vehicle_Utils(self.my_params)
        # self.state_moving = self.sim.get_linear_traj(self.steps_tot, self.my_params.T_s, speed=0.9)
        self.state_moving = self.sim.generate_target_trajectory(self.steps_tot, self.my_params.T_s, speed=0.75)

        self.camera_data = VisualTarget(start_state=self.state_moving[0,:], fov_h=self.my_params.fov_h, fov_v=self.my_params.fov_v, max_dist=10)
        self.seen_it_once = False
        self.wait_here = np.copy(self.state_now[0:12])
        self.camera_noise = np.random.normal(0, 0.00006, 3)
        # ROS Subscribers (Gazebo Truth)
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/model/bluerov2_heavy/odometry', # <--- Listening directly to Gazebo
            self.odom_callback,
            qos_sensor
        )

        # 8 separate publishers, one for each thruster
        self.thruster_pubs = []
        for i in range(1, 9):
            topic_name = f'/model/bluerov2_heavy/joint/thruster{i}_joint/cmd_thrust'
            pub = self.create_publisher(Float64, topic_name, 10)
            self.thruster_pubs.append(pub)

        #self.timer = self.create_timer(self.my_params.T_s, self.control_loop)
        
        self.get_logger().info("Direct Control Node Started. Waiting for Gazebo...")

        self.history_rov_x = []
        self.history_rov_y = []
        self.history_rov_z = []
        self.history_rov_ph = []
        self.history_rov_th = []
        self.history_rov_ps = []
        self.history_target = []


    def odom_callback(self, msg):
        """
        Convert ROS ENU Odometry to NMPC NED State
        """
        y = msg.pose.pose.position.y
        x = msg.pose.pose.position.x
        z = msg.pose.pose.position.z

        # Quaternions to rads
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        r = R.Rotation.from_quat([qx, qy, qz, qw])
        euler = r.as_euler('xyz', degrees=False)
        
        phi = euler[0]    # Roll
        theta = euler[1] # Pitch (inverted)
        psi = euler[2]  # Yaw
        
        u = msg.twist.twist.linear.x
        v = msg.twist.twist.linear.y
        w = msg.twist.twist.linear.z  #check sign?
        p = msg.twist.twist.angular.x
        q = msg.twist.twist.angular.y
        r = msg.twist.twist.angular.z

        self.state_now = np.array([x, y, z, phi, theta, psi, u, v, w, p, q, r])

        #   For plotting later on
        self.history_rov_x.append(x)
        self.history_rov_y.append(y)
        self.history_rov_z.append(z)
        self.history_rov_ph.append(phi)
        self.history_rov_th.append(theta)
        self.history_rov_ps.append(psi)

        self.history_target.append(self.state_moving[self.actual_step, 0:6])

        self.ekf.predict(self.u_previous)

        self.ekf.measurement_update(self.state_now)

        x_est = self.ekf.get_state_estimate()
        d_est = self.ekf.get_disturbance_estimate()

        # print("Disturbance estimate: ", d_est)

        # print("state now:", self.state_now)

        # print("state estimated:", x_est)

        self.camera_data.truth_update(self.state_moving[self.actual_step,0:6])
        if self.actual_step < (self.steps_tot-1):
            self.actual_step += 1
            print("target : ", self.state_moving[self.actual_step, 0:6])
        is_visible = self.camera_data.check_visibility(x_est[0:12], self.seen_it_once)
        if is_visible:
            # If visible modify the seen flag, estimate target position and get the trajectory
            self.seen_it_once = True
            est_target = self.camera_data.get_camera_estimate(x_est[0:12], dt = self.my_params.T_s, camera_noise = self.camera_noise)
            est_target_pos = est_target[0:3]
            est_target_vel = est_target[3:6]

            self.ref_target = get_shadow_traj(x_est[0:12], est_target_pos, est_target_vel, dt = self.my_params.T_s,horizon_N = self.my_params.N+1, desired_dist=2.5)
            #self.ref_target = get_shadow_ref(x_est[0:12], est_target_pos, est_target_vel, desired_dist=2.5)

        elif not is_visible and self.seen_it_once:

            # Visible before but not now: use last seen position and et imaginary reference based on last seen position
            est_target = self.camera_data.get_camera_estimate(x_est[0:12], dt = self.my_params.T_s, camera_noise = self.camera_noise)
            est_target_pos = est_target[0:3]
            est_target_vel = est_target[3:6]
            self.ref_target = get_shadow_ref(x_est[0:12], est_target_pos, est_target_vel, desired_dist=2.0)
                
            if self.camera_data.last_seen_t > 5.0:
                # If not seen for more than 5 seconds, just stay still
                self.ref_target = x_est[0:12]
                self.ref_target[6:12] = 0.0
                self.ref_target[4] = 0 
                self.seen_it_once = False
                self.wait_here = x_est[0:12]
        else:
            # Not visible and never seen: stay still till i see something
            self.ref_target = self.wait_here[0:12]


        u_optimal = self.solver.solve(x_est, self.ref_target, disturbance = d_est)
        # u_real = np.zeros(8)
        # for i in range(8):
        #     u_real[i] = np.sqrt(50*np.abs(u_optimal[i]))*np.sign(u_optimal[i]) 
        u_gazebo = np.zeros(8)
        u_gazebo[0] = u_optimal[0]
        u_gazebo[1] = u_optimal[1]
        u_gazebo[2] = u_optimal[2]
        u_gazebo[3] = u_optimal[3] 

        u_gazebo[4] = u_optimal[4]
        u_gazebo[5] = u_optimal[5] 
        u_gazebo[6] = u_optimal[6]
        u_gazebo[7] = u_optimal[7]
        print("u_optimal: ", u_gazebo)

        for i in range(8):
            msg = Float64()
            val = float(u_gazebo[i])
            msg.data = max(-35.0, min(35.0,val))
            self.thruster_pubs[i].publish(msg)

        self.u_previous = u_gazebo


#     def control_loop(self):

#         # EKF:: 
#         self.ekf.predict(self.u_previous)

#         if self.has_new_data == True:
#             self.has_new_data = False
#             self.ekf.measurement_update(self.state_now)

#         x_est = self.ekf.get_state_estimate()
#         d_est = self.ekf.get_disturbance_estimate()

#         u_optimal = self.solver.solve(x_est, self.ref_target, disturbance = d_est)
#         print("u_optimal: ", u_optimal)

#         u_gazebo = np.zeros(8)
        
    
#         u_gazebo[0] = u_optimal[0] 
#         u_gazebo[1] = u_optimal[1]
#         u_gazebo[2] = u_optimal[2]
#         u_gazebo[3] = u_optimal[3]

#         u_gazebo[4] = u_optimal[4]  
#         u_gazebo[5] = u_optimal[5]
#         u_gazebo[6] = u_optimal[6]
#         u_gazebo[7] = u_optimal[7]

#         # u_gazebo[0] = 0.0 # Up Right CW
#         # u_gazebo[1] = 0.0 # Up Left CCW
#         # u_gazebo[2] = 0.0 # Down Right CCW
#         # u_gazebo[3] = 0.0 # Down left makes it spin CW
#         # u_gazebo[4] = 0.0 #Up Right goes down?????
#         # u_gazebo[5] = 10.0 #Up Left goes underwater ?
#         # u_gazebo[6] = 10.0 #Down Right goes underwater
#         # u_gazebo[7] = 0.0 #Down left goes underwater

# # 1 e 3 hanno x, pitch, yaw opposti
# # 1 e 3 hanno stessa Y,roll: positiva
# # 0 e 2 hanno stesso Y,roll : negativo
# # 0 e 2 hanno X, pitch, yaw opposti
# # 0 e 1 hanno x NEG, pitch pos
# # 0 e 3 hanno stesso yaw: neg, il resto uguale



def main(args=None):
    rclpy.init(args=args)
    node = BlueROVMPC()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        zero_msg = Float64()
        zero_msg.data = 0.0
        # for pub in node.thruster_pubs:
        #     pub.publish(zero_msg)
        rov_x = np.array(node.history_rov_x)
        rov_y = np.array(node.history_rov_y)
        rov_z = np.array(node.history_rov_z)
        rov_ph = np.array(node.history_rov_ph)
        rov_th = np.array(node.history_rov_th)
        rov_ps = np.array(node.history_rov_ps)
        target_data = np.array(node.history_target)
        dt = node.my_params.T_s

        node.sim.get_error_avg_std([rov_x, rov_y, rov_z], node.state_moving[:,:3].T, target_data[:,:3].T)
        LOS_plot_camera_fov(rov_x, rov_y, rov_z, rov_ps, rov_th, node.state_moving, dt)

        if len(rov_x) > 0:
            slider, fig = LOS_interactive_viewer(rov_x, rov_y, rov_z, target_data, dt)
            plt.show()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()