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

# ROS / MavROS msgs
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64  
from geometry_msgs.msg import Point
from mavros_msgs.msg import OverrideRCIn, State
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

# ---------------------------------------------------------------------------
# USER FLAGS — change these before each test
# ---------------------------------------------------------------------------

# CHANGE A: Set False to bypass AEKFD and feed raw GPS/odom state to the NMPC.
#           Useful to isolate EKF issues during first hardware tests.
USE_EKF = True

# Max thruster force [N] that maps to ±PWM_RANGE. Tune to your thruster curve.
MAX_THRUSTER_FORCE_N = 40.0

# CHANGE B: PWM range around neutral (1500). ±100 = ~25% throttle for first tests.
#           Increase gradually once behaviour is verified (max meaningful value: 400).
PWM_RANGE = 100

# Max estimated speed [m/s] before the safety guard cuts thrust.
MAX_SPEED_MS = 3.0

# Odom watchdog timeout [s]: releases overrides if no odom message arrives.
ODOM_WATCHDOG_TIMEOUT_S = 1.0

# ---------------------------------------------------------------------------

class BlueBoatMPC(Node):
    def __init__(self):
        super().__init__('blueboat_nmpc_node')

        self.my_params = BlueBoat_Params()
        self.get_logger().info("Initilizing Acados solver...")
        self.solver = Acados_Solver_Wrapper(self.my_params)
        self.get_logger().info("Solver Ready!")
        acados_model = export_vehicle_model(self.my_params)

        # CHANGE A: EKF is always initialised (we need its output shape for d_est),
        # but it is only updated/used when USE_EKF=True.
        self.ekf = AEKFD(acados_model, self.my_params)

        self.u_previous = np.zeros(self.my_params.nu)
        self.state_now  = np.zeros(self.my_params.nx)

        self.ekf.set_state_estimate(self.state_now)
        self.ref_target = np.zeros(12)

        self.steps_tot  = 1000
        self.actual_step = 0
        self.sim = Vehicle_Utils(self.my_params)

        self.state_moving = self.sim.get_linear_traj(self.steps_tot, self.my_params.T_s, speed=0.75)

        self.camera_data = VisualTarget(start_state=self.state_moving[0,:], fov_h=self.my_params.fov_h, fov_v=self.my_params.fov_v, max_dist=20)
        self.seen_it_once = False
        self.wait_here    = np.copy(self.state_now[0:12])
        self.camera_noise = np.random.normal(0, 0.000, 3)

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Odometry Subscription
        self.odom_sub = self.create_subscription(
            Odometry,
            '/mavros/local_position/odom', 
            self.odom_callback,
            qos_sensor
        )
        
        # RC Override Publisher
        self.rc_pub = self.create_publisher(
            OverrideRCIn,
            '/mavros/rc/override',
            10
        )
        
        # MAVROS State & Services
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, 10)
        self.current_state  = None
        self.state_received = False
        
        self.arm_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # Camera
        self.use_real_camera      = True  
        self.real_target_relative = np.zeros(3)
        self.last_camera_update   = 0.0
        
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

        # Odom watchdog
        self.last_odom_update = time.time()
        self.watchdog_timer   = self.create_timer(0.5, self.odom_watchdog_callback)

        if USE_EKF:
            self.get_logger().info('State estimator: AEKFD')
        else:
            self.get_logger().info('State estimator: RAW GPS/odom (EKF bypassed)')

        self.get_logger().info('BlueBoat is ready')

        # History arrays for plotting/saving
        self.history_rov_x  = []
        self.history_rov_y  = []
        self.history_rov_z  = []
        self.history_rov_ph = []
        self.history_rov_th = []
        self.history_rov_ps = []
        self.history_target = []   # estimated target pose, updated every step
        self.history_ref    = []
        self.thrust_history = []
        self.history_real_target_x = []
        self.history_real_target_y = []
        self.history_real_target_z = []
        self.ref_x = []
        self.ref_y = []
        self.ref_z = []

    # --- Hardware Safety & State Methods ---
    
    def state_cb(self, msg):
        self.current_state  = msg
        self.state_received = True

    def odom_watchdog_callback(self):
        if time.time() - self.last_odom_update > ODOM_WATCHDOG_TIMEOUT_S:
            self.get_logger().warn(
                f'WATCHDOG: No odometry for >{ODOM_WATCHDOG_TIMEOUT_S}s! Releasing overrides.',
                throttle_duration_sec=2.0
            )
            self.release_overrides()

    def wait_for_mode(self, target_mode, timeout=5.0):
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_state and self.current_state.mode == target_mode:
                return True
        return False

    def wait_for_armed(self, armed=True, timeout=5.0):
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_state and self.current_state.armed == armed:
                return True
        return False

    def arm_gentle(self):
        while not self.state_received and rclpy.ok():
            self.get_logger().info('Waiting for MAVROS state topic...')
            rclpy.spin_once(self, timeout_sec=0.5)

        self.get_logger().info('Setting mode to MANUAL...')
        req_mode = SetMode.Request()
        req_mode.custom_mode = 'MANUAL'
        future_mode = self.mode_client.call_async(req_mode)
        rclpy.spin_until_future_complete(self, future_mode, timeout_sec=2.0)
        
        if not future_mode.result() or not future_mode.result().mode_sent:
            self.get_logger().error('Failed to send MANUAL mode command')
            return False

        if not self.wait_for_mode('MANUAL'):
            self.get_logger().error('Mode did not switch to MANUAL')
            return False

        self.get_logger().info('Arming with multiple attempts...')
        req_arm = CommandBool.Request()
        req_arm.value = True
        armed = False

        for attempt in range(4): 
            self.get_logger().info(f'Arming attempt {attempt+1}/4')
            future_arm = self.arm_client.call_async(req_arm)
            rclpy.spin_until_future_complete(self, future_arm, timeout_sec=2.0)
            
            if future_arm.result() and future_arm.result().success:
                self.get_logger().info(f'Attempt {attempt+1} service reported success')
            else:
                self.get_logger().warn(f'Attempt {attempt+1} service did not report success')

            time.sleep(0.5)

            if self.current_state and self.current_state.armed:
                armed = True
                break

        if not armed:
            if self.wait_for_armed(True, timeout=5.0):
                armed = True

        if armed:
            self.get_logger().info('Armed successfully')
            return True
        else:
            self.get_logger().error('Vehicle did not become armed')
            return False

    def disarm_gentle(self):
        self.get_logger().info('Disarming vehicle...')
        req_arm = CommandBool.Request()
        req_arm.value = False
        
        for attempt in range(4):
            future_arm = self.arm_client.call_async(req_arm)
            rclpy.spin_until_future_complete(self, future_arm, timeout_sec=2.0)
            time.sleep(0.5)
            
            if self.current_state and not self.current_state.armed:
                self.get_logger().info('Verified disarmed via state topic.')
                return True
                
        self.get_logger().error('CRITICAL: Vehicle refused to disarm!')
        return False

    def release_overrides(self):
        self.get_logger().info('Releasing RC overrides to neutral...')
        msg = OverrideRCIn()
        msg.channels = [65535] * 18 
        self.rc_pub.publish(msg)

    # --- Control & NMPC Methods ---

    def force_to_pwm(self, force_n):
        """Convert a force [N] to PWM, scaled to PWM_RANGE around neutral 1500."""
        force_clipped = np.clip(force_n, -MAX_THRUSTER_FORCE_N, MAX_THRUSTER_FORCE_N)
        # CHANGE B: uses PWM_RANGE constant (100 for first tests) instead of hardcoded 400.
        pwm = int(1500 + (force_clipped / MAX_THRUSTER_FORCE_N) * PWM_RANGE)
        return int(np.clip(pwm, 1100, 1900))  # absolute hard saturation guard

    def move(self, force_left, force_right):
        """
        Convert independent L/R NMPC forces [N] to ArduPilot steering+throttle PWM.

        CHANGE C: restored steering+throttle mapping to match ArduPilot MANUAL mode,
        as verified working in blueboat_labTest.py. ArduPilot does the differential
        motor mixing internally.
          channel 1 (index 0) = steering  (yaw demand)
          channel 3 (index 2) = throttle  (forward demand)
        """
        forward_force = (force_left + force_right) / 2.0  # mean  → throttle
        turn_force    = (force_left - force_right) / 2.0  # diff  → steering

        pwm_throttle = self.force_to_pwm(forward_force)
        pwm_steering = self.force_to_pwm(turn_force)

        msg = OverrideRCIn()
        channels = [65535] * 18
        channels[0] = pwm_steering  # channel 1 — steering
        channels[2] = pwm_throttle  # channel 3 — throttle
        msg.channels = channels
        self.rc_pub.publish(msg)

    def get_state_now(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        r = R.Rotation.from_quat([qx, qy, qz, qw])
        euler = r.as_euler('xyz', degrees=False)
        
        phi   = euler[0]
        theta = euler[1]
        psi   = euler[2]
        
        u     = msg.twist.twist.linear.x
        v     = msg.twist.twist.linear.y
        w     = msg.twist.twist.linear.z  
        p     = msg.twist.twist.angular.x
        q     = msg.twist.twist.angular.y
        r_ang = msg.twist.twist.angular.z

        self.state_now = np.array([x, y, z, phi, theta, psi, u, v, w, p, q, r_ang])
        return self.state_now
    
    def vision_callback(self, msg):
        self.real_target_relative = np.array([msg.x, msg.y, msg.z])
        self.last_camera_update   = time.time()
        self.seen_it_once         = True

    def update_target_simulated(self, x_est):
        self.camera_data.truth_update(self.state_moving[self.actual_step, 0:6])
        
        if self.actual_step < (self.steps_tot - 1):
            self.actual_step += 1

        est_target_pos = np.zeros(3)  # default; keeps arrays aligned even if not visible
            
        is_visible = self.camera_data.check_visibility(self.state_now, self.seen_it_once)
        
        if is_visible:
            self.seen_it_once = True
            est_target     = self.camera_data.get_camera_estimate(x_est[0:12], dt=self.my_params.T_s, camera_noise=self.camera_noise)
            est_target_pos = est_target[0:3]
            est_target_vel = est_target[3:6]
            self.ref_target = get_shadow_LOS(x_est[0:12], est_target_pos, est_target_vel, desired_dist=2.5)
            
        elif not is_visible and self.seen_it_once:
            est_target     = self.camera_data.get_camera_estimate(x_est[0:12], dt=self.my_params.T_s, camera_noise=self.camera_noise)
            est_target_pos = est_target[0:3]
            est_target_vel = est_target[3:6]
            self.ref_target = get_shadow_LOS(x_est[0:12], est_target_pos, est_target_vel, desired_dist=1.0)

            if self.camera_data.last_seen_t > 30.0:
                self.ref_target        = x_est[0:12]
                self.ref_target[6:12]  = 0.0
                self.ref_target[4]     = 0 
                self.seen_it_once      = False
                self.wait_here         = x_est[0:12]
        else:
            self.ref_target = self.wait_here[0:12]

        # Always append to keep array sizes consistent with odom arrays.
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
            
            cam_dx   = self.real_target_relative[0]
            cam_dist = self.real_target_relative[2]
            
            target_world_x = boat_pos[0] + (cam_dist * np.cos(boat_yaw)) - (cam_dx * np.sin(boat_yaw))
            target_world_y = boat_pos[1] + (cam_dist * np.sin(boat_yaw)) + (cam_dx * np.cos(boat_yaw))
            target_world_z = boat_pos[2] 
            
            est_target_pos  = np.array([target_world_x, target_world_y, target_world_z])
            self.ref_target = get_shadow_LOS(x_est[0:12], est_target_pos, est_target_vel, desired_dist=2.5)

        elif not is_actually_visible and self.seen_it_once:
            self.ref_target = get_shadow_LOS(x_est[0:12], est_target_pos, est_target_vel, desired_dist=1.0)
            
            if (time.time() - self.last_camera_update) > 5.0:
                self.ref_target       = x_est[0:12]
                self.ref_target[6:12] = 0.0
                self.seen_it_once     = False
                self.wait_here        = x_est[0:12]
        else:
            self.ref_target = self.wait_here[0:12]
             
        # Always append to keep array sizes consistent with odom arrays.
        self.history_real_target_x.append(est_target_pos[0])
        self.history_real_target_y.append(est_target_pos[1])
        self.history_real_target_z.append(est_target_pos[2])

    def odom_callback(self, msg):
        # Reset watchdog timestamp.
        self.last_odom_update = time.time()

        ros_state = self.get_state_now(msg)
        
        self.history_rov_x.append(ros_state[0])
        self.history_rov_y.append(ros_state[1])
        self.history_rov_z.append(ros_state[2])
        self.history_rov_ph.append(ros_state[3])
        self.history_rov_th.append(ros_state[4])
        self.history_rov_ps.append(ros_state[5])

        # CHANGE A: choose state estimate source based on USE_EKF flag.
        if USE_EKF:
            self.ekf.predict(self.u_previous)
            self.ekf.measurement_update(self.state_now)
            x_est = self.ekf.get_state_estimate()
            d_est = self.ekf.get_disturbance_estimate()
        else:
            # Raw GPS/odom path: bypass EKF entirely.
            # d_est is zeroed — the NMPC will receive no disturbance feedforward.
            x_est = self.state_now.copy()
            d_est = np.zeros_like(self.ekf.get_disturbance_estimate())

        print("position : ", x_est[0:6])

        # Speed safety guard (works identically for EKF and raw state path).
        speed = np.linalg.norm(x_est[6:9])
        if speed > MAX_SPEED_MS:
            self.get_logger().error(
                f'SAFETY: Estimated speed {speed:.2f} m/s exceeds limit {MAX_SPEED_MS} m/s! '
                'Releasing overrides.'
            )
            self.release_overrides()
            return

        if self.use_real_camera:
            self.update_target_real(x_est)
        else:
            self.update_target_simulated(x_est)

        # CHANGE D (plot fix): log the actual estimated target position every step,
        # instead of state_moving[actual_step] which stays frozen at step 0 in real mode.
        self.history_target.append(np.array([
            self.history_real_target_x[-1],
            self.history_real_target_y[-1],
            self.history_real_target_z[-1],
            0.0, 0.0, x_est[5]   # pad phi/theta to 0, use boat yaw as proxy for psi
        ]))

        ref_safe = np.atleast_2d(self.ref_target)  
        self.ref_x.append(ref_safe[0, 0])
        self.ref_y.append(ref_safe[0, 1])
        self.ref_z.append(ref_safe[0, 2])

        u_optimal = self.solver.solve(x_est, self.ref_target, disturbance=d_est)
        
        self.move(u_optimal[0], u_optimal[1])
        self.u_previous = u_optimal
        self.thrust_history.append(u_optimal)


def main(args=None):
    rclpy.init(args=args)
    node = BlueBoatMPC()
    
    # Arm the boat safely BEFORE spinning the node.
    success = node.arm_gentle()
    if not success:
        node.get_logger().error("Failed to arm the vehicle. Aborting mission.")
        node.destroy_node()
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Safely stopping...")
    finally:
        # Release thrust and disarm before anything else.
        node.release_overrides()
        time.sleep(0.5) 
        node.disarm_gentle()

        # Collect all data arrays.
        rov_x  = np.array(node.history_rov_x)
        rov_y  = np.array(node.history_rov_y)
        rov_z  = np.array(node.history_rov_z)
        rov_ph = np.array(node.history_rov_ph)
        rov_th = np.array(node.history_rov_th)
        rov_ps = np.array(node.history_rov_ps)
        target_data        = np.array(node.history_target)
        real_target_data_x = np.array(node.history_real_target_x)
        real_target_data_y = np.array(node.history_real_target_y)
        real_target_data_z = np.array(node.history_real_target_z)
        ref_data_x     = np.array(node.ref_x)
        ref_data_y     = np.array(node.ref_y)
        ref_data_z     = np.array(node.ref_z)
        thrust_history = np.array(node.thrust_history)
        dt = node.my_params.T_s

        # Robust min-length truncation across ALL dynamic arrays.
        lengths = [len(rov_x), len(ref_data_x), len(thrust_history),
                   len(node.state_moving), len(real_target_data_x)]
        if min(lengths) == 0:
            print("No data recorded. Shutting down safely.")
            node.destroy_node()
            rclpy.shutdown()
            return

        min_len = min(lengths)

        rov_x  = rov_x[:min_len]
        rov_y  = rov_y[:min_len]
        rov_z  = rov_z[:min_len]
        rov_ph = rov_ph[:min_len]
        rov_th = rov_th[:min_len]
        rov_ps = rov_ps[:min_len]
        target_data        = target_data[:min_len, :]
        real_target_data_x = real_target_data_x[:min_len]
        real_target_data_y = real_target_data_y[:min_len]
        real_target_data_z = real_target_data_z[:min_len]
        ref_data_x     = ref_data_x[:min_len]
        ref_data_y     = ref_data_y[:min_len]
        ref_data_z     = ref_data_z[:min_len]
        thrust_history = thrust_history[:min_len]

        # CHANGE D (plot fix): in real camera mode the "target trajectory" shown in
        # the plots is the estimated real target, not the dummy state_moving trajectory
        # (which stays fixed at origin in real mode). In simulated mode, state_moving
        # is still correct.
        if node.use_real_camera:
            plot_target_x    = real_target_data_x
            plot_target_y    = real_target_data_y
            plot_target_z    = real_target_data_z
            # LOS_plot_camera_fov needs a (N,6) array: x,y,z,phi,theta,psi.
            plot_target_traj = np.column_stack([
                real_target_data_x, real_target_data_y, real_target_data_z,
                np.zeros(min_len), np.zeros(min_len), rov_ps
            ])
        else:
            state_moving_sliced = node.state_moving[:min_len, :]
            plot_target_x    = state_moving_sliced[:, 0]
            plot_target_y    = state_moving_sliced[:, 1]
            plot_target_z    = state_moving_sliced[:, 2]
            plot_target_traj = state_moving_sliced

        save_path = os.path.join(current_dir, 'blueboat_flight_data.npz')
        np.savez(save_path, 
                 rov_x=rov_x, rov_y=rov_y, rov_z=rov_z, 
                 rov_ph=rov_ph, rov_th=rov_th, rov_ps=rov_ps, 
                 target=target_data,
                 ref_x=ref_data_x, ref_y=ref_data_y, ref_z=ref_data_z,
                 real_target_data_x=real_target_data_x,
                 real_target_data_y=real_target_data_y,
                 real_target_data_z=real_target_data_z,
                 thrust=thrust_history, dt=dt)

        plot_TT_3d(
            plot_target_x, plot_target_y, plot_target_z,
            ref_data_x, ref_data_y, ref_data_z,
            rov_x, rov_y, rov_z,
            rov_ph, rov_th, rov_ps,
            thrust_history, dt
        )

        LOS_plot_camera_fov(rov_x, rov_y, rov_z, rov_ps, rov_th, plot_target_traj, dt)

        if len(rov_x) > 0:
            plt.show()
            
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()