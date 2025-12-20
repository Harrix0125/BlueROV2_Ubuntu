import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from mavros_msgs.msg import OverrideRCIn

import numpy as np
import sys
import os

#sorry but here i have my blueROV2 repo and i need to add it to the path... the NMPC is there

sys.path.append(os.path.expanduser('~/ros2_ws/src/mpc_controller/BlueROV2'))

from nmpc_solver_acados import Acados_Solver_Wrapper
from nmpc_params import NMPC_params as MPCC

class NMPCNode(Node):
    def __init__(self):
        super().__init__('mpc_node')

        self.get_logger().info("Initializing Acados Solver...")
        self.solver = Acados_Solver_Wrapper()

        self.current_state = np.zeros(12)  # [x, y, z, phi, theta, psi, u, v, w, p, q, r]
        self.target_state = np.zeros(12)
        # Example Target: 2 meters deep, facing forward
        self.target_state[2] = -2.0  # Depth Z
        self.target_state[5] = 0.0  # Yaw Psi

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability = DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        #self.create_subscription(
        #    Odometry,
        #    '/mavros/bluerov2_heavy/odometry',
        #    self.odom_callback,
        #    qos_sensor
        #)

        self.create_subscription(
            Odometry,
            '/model/bluerov2_heavy/odometry',
            self.odom_callback,
            qos_sensor
        )
        

        self.thruster_pubs = []
        for i in range(1, 9):
            topic = f'/model/bluerov2_heavy/joint/thruster{i}_joint/cmd_thrust'
            pub = self.create_publisher(Float64, topic, 10)
            self.thruster_pubs.append(pub)
        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)

        self.dt = MPCC.T_s
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("NMPC Node Initialized. Waiting for odometry data...")

    def odom_callback(self, msg):
        """
        Updates the current state vector [12x1] from ROS Odometry.
        ROS Odom: Position (x,y,z), Orientation (Quat), Linear Vel (u,v,w), Angular Vel (p,q,r)
        """
        print("Odom callback triggered")
        # Position
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = msg.pose.pose.position.y
        self.current_state[2] = msg.pose.pose.position.z

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        # Orientation (Quaternion -> Euler)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.current_state[3], self.current_state[4], self.current_state[5] = self.euler_from_quaternion(qx, qy, qz, qw)

        # Linear Velocity (Body Frame)
        self.current_state[6] = msg.twist.twist.linear.x
        self.current_state[7] = msg.twist.twist.linear.y
        self.current_state[8] = msg.twist.twist.linear.z

        # angular Velocity (Body Frame)
        self.current_state[9] = msg.twist.twist.angular.x
        self.current_state[10] = msg.twist.twist.angular.y
        self.current_state[11] = msg.twist.twist.angular.z

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)
        """
        print("Converting quaternion to euler angles")
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z
    
    def map_thrust_to_pwm(self, thrust):
        """
        Maps thrust command (N) to PWM signal for BlueROV2 thrusters.
        Assuming linear mapping for simplicity.
        """
        print("Mapping thrust to PWM")
        # Example mapping parameters (to be adjusted based on actual thruster characteristics)
        min_thrust = -25.0  # N
        max_thrust = 25.0   # N
        min_pwm = 1100      # PWM signal for full reverse
        max_pwm = 1900      # PWM signal for full forward
        neutral_pwm = 1500  # PWM signal for neutral

        # Saturate thrust just in case
        if thrust < min_thrust:
            thrust = min_thrust
        elif thrust > max_thrust:
            thrust = max_thrust
        
        pwm = neutral_pwm + (thrust / max_thrust) * (max_pwm - neutral_pwm)

        return int(pwm)

    def control_loop(self):
        """
        Main control loop: Solve NMPC and publish thruster commands.
        """
        print("Control loop triggered")
        # Solve NMPC
        try:
            u_opt = self.solver.solve(self.current_state, self.target_state)
        except Exception as e:
            self.get_logger().warn(f"NMPC Solver failed: {e}")
            u_opt = np.zeros(8)
            return
        

        msg = OverrideRCIn()
        msg.channels = [65535]*8
        # Publish thruster commands
        for i, thrust in enumerate(u_opt):
            msg = Float64()
            msg.data = float(thrust)
            self.thruster_pubs[i].publish(msg)
        #for i in range(8):
        #    if i < len(u_opt):
        #        msg.channels[i] = self.map_thrust_to_pwm(u_opt[i])
        #self.rc_pub.publish(msg)

#def main(args=None):
#        rclpy.init(args=args)
#        Node = NMPCNode()
#        rclpy.spin(Node)
#        Node.destroy_node()
#        rclpy.shutdown()
#change
def main(args=None):
    rclpy.init(args=args)
    node = NMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # SAFETY: Stop motors on shutdown
        stop_msg = OverrideRCIn()
        stop_msg.channels = [1500] * 8 + [65535] * 10
        node.rc_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
        main()

