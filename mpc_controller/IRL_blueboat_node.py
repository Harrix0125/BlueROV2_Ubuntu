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

# ROS / MavROS msgs and stuff cmon
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64  
from geometry_msgs.msg import Point
from mavros_msgs.msg import OverrideRCIn
from mavros_msgs.srv import CommandBool, SetMode

# Imports for NMPC
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

        #   This listens to blueboat odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/mavros/local_position/odom', 
            self.odom_callback,
            qos_sensor
        )
        #   Publishes (obv) the RC
        self.rc_pub = self.create_publisher(
            OverrideRCIn,
            '/mavros/rc/override',
            10
        )
        #   MavROS commands
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_boat()

        #   Camera Node
        self.use_real_camera = True  # Set to False to use the simulated imaginary target
        self.real_target_relative = np.zeros(3)
        self.last_camera_update = 0.0
        
        if self.use_real_camera:
            from geometry_msgs.msg import Point
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

        self.history_rov_x = []
        self.history_rov_y = []
        self.history_rov_z = []
        self.history_rov_ph = []
        self.history_rov_th = []
        self.history_rov_ps = []
        self.history_target = []
        self.history_ref = []
        self.thrust_history = []
        self.history_real_target = []
        self.ref_x = []
        self.ref_y = []
        self.ref_z = []


    def move(self, sx_speed, dx_speed):
        """
        Translates NMPC left/right outputs to MAVROS PWM RC Overrides.
        Note: Ensure sx_speed and dx_speed are normalized between -1.0 and 1.0.
        """
        forward = (sx_speed + dx_speed) / 2.0
        turn = (sx_speed - dx_speed) / 2.0  

        #   Convert to PWM signals (1500 is neutral) : 
        #       Using 400 as a multiplier, max speed is 1900, min is 1100
        pwm_throttle = int(1500 + (forward * 400))
        pwm_steering = int(1500 + (turn * 400))

        pwm_throttle = max(1100, min(1900, pwm_throttle))
        pwm_steering = max(1100, min(1900, pwm_steering))

        # Ok i found this i hope this works
        msg = OverrideRCIn()
        channels = [65535] * 18
        channels[0] = pwm_steering 
        channels[2] = pwm_throttle  
        
        msg.channels = channels
        self.rc_pub.publish(msg)

    def get_state_now(self, msg):
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

        return self.state_now
    
    def vision_callback(self, msg):
        """Saves the latest AprilTag position from the camera"""
        self.real_target_relative = np.array([msg.x, msg.y, msg.z])
        self.last_camera_update = time.time()
        self.seen_it_once = True

    def update_target_simulated(self, x_est):
        """Calculates the NMPC reference target using the imaginary moving target."""
        self.camera_data.truth_update(self.state_moving[self.actual_step, 0:6])
        
        if self.actual_step < (self.steps_tot-1):
            self.actual_step += 1
            
        is_visible = self.camera_data.check_visibility(self.state_now, self.seen_it_once)
        
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


    def update_target_real(self, x_est):
        """Calculates the NMPC reference target using the physical stereo camera."""
        # Check if we received a fresh camera frame in the last 1.0 seconds
        is_actually_visible = (time.time() - self.last_camera_update) < 1.0

        # We need default estimates to pass to the LOS function if we lose the tag
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
            
            # Keep a distance of 2.5 meters from the real board
            self.ref_target = get_shadow_LOS(x_est[0:12], est_target_pos, est_target_vel, desired_dist=2.5)

        elif not is_actually_visible and self.seen_it_once:
            # Lost tag! Fall back to last known position but close the distance safely
            self.ref_target = get_shadow_LOS(x_est[0:12], est_target_pos, est_target_vel, desired_dist=1.0)
            
            if (time.time() - self.last_camera_update) > 5.0:
                # If not seen for more than 5 seconds, station-keep
                self.ref_target = x_est[0:12]
                self.ref_target[6:12] = 0.0
                self.seen_it_once = False
                self.wait_here = x_est[0:12]
        else:
             # Never seen it, wait here
             self.ref_target = self.wait_here[0:12]
        self.history_real_target.append(est_target_pos.copy())

    def odom_callback(self, msg):
        """
        Convert ROS ENU Odometry to NMPC NED State
        """
        ros_state = self.get_state_now(msg)
        #   For plotting later on
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
        # Publishing the controls
        self.move(u_optimal[0], u_optimal[1])
        self.u_previous = u_optimal
        self.thrust_history.append(u_optimal)

    def arm_boat(self):
        """Automatically sets the boat to MANUAL mode and ARMS the thrusters."""
        self.get_logger().info("Waiting for MAVROS arming services...")
        self.arm_client.wait_for_service()
        self.mode_client.wait_for_service()
        
        req_mode = SetMode.Request()
        req_mode.custom_mode = "MANUAL"
        self.mode_client.call_async(req_mode)
        
        req_arm = CommandBool.Request()
        req_arm.value = True
        self.arm_client.call_async(req_arm)
        self.get_logger().info("BlueBoat Armed and Ready!")

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
        # for pub in node.thruster_pubs:
        #     pub.publish(zero_msg)
        rov_x = np.array(node.history_rov_x)
        rov_y = np.array(node.history_rov_y)
        rov_z = np.array(node.history_rov_z)
        rov_ph = np.array(node.history_rov_ph)
        rov_th = np.array(node.history_rov_th)
        rov_ps = np.array(node.history_rov_ps)
        target_data = np.array(node.history_target)
        real_target_data = np.array(node.history_real_target)
        ref_data_x = np.array(node.ref_x)
        ref_data_y = np.array(node.ref_y)
        ref_data_z = np.array(node.ref_z)
        node.thrust_history = np.array(node.thrust_history)

        dt = node.my_params.T_s
        size_max = min(np.size(node.state_moving[:,0]), np.size(rov_x))-1
        
        save_path = os.path.join(current_dir, 'blueboat_flight_data.npz')
        np.savez(save_path, 
                        rov_x=rov_x, rov_y=rov_y, rov_z=rov_z, 
                        rov_ph=rov_ph, rov_th=rov_th, rov_ps=rov_ps, 
                        target=target_data, ref_x=ref_data_x, 
                        real_target=real_target_data,
                        ref_y=ref_data_y, ref_z=ref_data_z, 
                        thrust=node.thrust_history, dt=dt)

        
        # node.sim.get_error_avg_std([rov_x, rov_y, rov_z], node.state_moving[:,:3].T, target_data[:,:3].T)
        plot_TT_3d(np.array(node.state_moving[:size_max+1,0]), np.array(node.state_moving[:size_max+1,1]), np.array(node.state_moving[:size_max+1,2]),
            ref_data_x, ref_data_y, ref_data_z, # Reference
            np.array(rov_x[:size_max]), np.array(rov_y[:size_max]), np.array(rov_z[:size_max]),    # ROV Position
            np.array(rov_ph[:size_max]), np.array(rov_th[:size_max]), np.array(rov_ps[:size_max]), # ROV Angles
            node.thrust_history, dt
        )
        # LOS_plot_camera_fov(rov_x, rov_y, rov_z, rov_ps, rov_th, node.state_moving, dt)
        # LOS_plot_dynamics(rov_x[:size_max], rov_y[:size_max], rov_z[:size_max], node.state_moving[:size_max+1,:], dt, desired_dist=2.0)

        LOS_plot_camera_fov(rov_x[:size_max], rov_y[:size_max], rov_z[:size_max], rov_ps[:size_max], rov_th[:size_max], node.state_moving[:size_max+1,:], dt)

        if len(rov_x) > 0:
            # slider, fig = LOS_interactive_viewer(rov_x, rov_y, rov_z, target_data, dt)
            plt.show()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


    