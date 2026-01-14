import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# ROS Msgs (Direct Control)
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64  

# NMPC Imports
import numpy as np
import scipy.spatial.transform.rotation as R

# Relative imports (keep these as they were working)
from .BlueROV2.nmpc_params import NMPC_params as MPCC
from .BlueROV2.nmpc_solver_acados import Acados_Solver_Wrapper

class BlueROVMPC(Node):
    def __init__(self):
        super().__init__('bluerov_nmpc_node')

        # Initialize NMPC Solver
        self.get_logger().info("Initializing Acados Solver...")
        self.solver = Acados_Solver_Wrapper()
        self.get_logger().info("Solver Ready!")

        # State & Reference
        self.state_now = np.zeros(12) 
        self.state_now[2] = -2
        self.ref_target = np.zeros(12)
        self.ref_target[2] = -5.0  
        
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

        # ROS Publishers (Direct Force Control)
        # We need 8 separate publishers, one for each thruster
        self.thruster_pubs = []
        for i in range(1, 9):
            topic_name = f'/model/bluerov2_heavy/joint/thruster{i}_joint/cmd_thrust'
            pub = self.create_publisher(Float64, topic_name, 10)
            self.thruster_pubs.append(pub)

        # Control Loop Timer
        self.timer = self.create_timer(MPCC.T_s, self.control_loop)
        
        self.get_logger().info("Direct Control Node Started. Waiting for Gazebo...")

    def odom_callback(self, msg):
        """
        Convert ROS ENU Odometry to NMPC NED State
        """
        # ROS ENU (East-North-Up) -> NED (North-East-Down)
        # x_ned = y_enu
        # y_ned = x_enu
        # z_ned = z_enu
        x = msg.pose.pose.position.y
        y = msg.pose.pose.position.x
        z = msg.pose.pose.position.z

        # Orientation
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        r = R.Rotation.from_quat([qx, qy, qz, qw])
        euler = r.as_euler('xyz', degrees=False)
        
        phi = euler[0]    # Roll
        theta = -euler[1] # Pitch (inverted)
        psi = -euler[2] + np.pi/2 # Yaw
        
        # Velocity
        u = msg.twist.twist.linear.x
        v = -msg.twist.twist.linear.y
        w = msg.twist.twist.linear.z  #check sign?
        p = msg.twist.twist.angular.x
        q = -msg.twist.twist.angular.y
        r = -msg.twist.twist.angular.z

        self.state_now = np.array([x, y, z, phi, theta, psi, u, v, w, p, q, r])

    def control_loop(self):
        # Solve NMPC (Returns Force in Newtons)
        u_optimal = self.solver.solve(self.state_now, self.ref_target)

        # Publish Directly to Gazebo
        # u_optimal should be an array of 8 floats (Newtons)
        #for i in range(8):
        #    force_msg = Float64()
        #    
        #    # Safety
        #    force_val = float(u_optimal[i])
        #    force_val = max(-35.0, min(35.0, force_val))
        #    
        #    force_msg.data = force_val
        #    
        #    # Publish to thruster i+1
        #    self.thruster_pubs[i].publish(force_msg)

        for i in range(8):
            force_msg = Float64()
            u_optimal[i] = 0
            force_val = float(35)
            zerooo = float(u_optimal[i])

            force_msg.data = zerooo
            # Publish to thruster i+1
            if i == 1:
                force_msg.data = force_val

            self.thruster_pubs[i].publish(force_msg)



def main(args=None):
    rclpy.init(args=args)
    node = BlueROVMPC()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Safety: Send 0 thrust on exit
        zero_msg = Float64()
        zero_msg.data = 0.0
        for pub in node.thruster_pubs:
            pub.publish(zero_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()