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
from .BlueROV2.nmpc_params import BlueROV_Params, BlueBoat_Params
from .BlueROV2.nmpc_solver_acados import Acados_Solver_Wrapper
from .BlueROV2.AEKFD import AEKFD
from .BlueROV2.model import export_vehicle_model

class BlueROVMPC(Node):
    def __init__(self):
        super().__init__('bluerov_nmpc_node')

        self.my_params = BlueROV_Params()
        self.get_logger().info("Initializing Acados Solver...")
        self.solver = Acados_Solver_Wrapper(self.my_params)
        self.get_logger().info("Solver Ready!")
        acados_model = export_vehicle_model(self.my_params)
        self.ekf = AEKFD(acados_model, self.my_params)

        self.u_previous = np.zeros(8)

        # State & Reference
        self.state_now = np.zeros(12) 
        self.state_now[2] = -2
        self.ekf.set_state_estimate(self.state_now)

        self.ref_target = np.zeros(12)
        self.ref_target[2] = -2.0  
        self.ref_target[1] = 0.0  
        self.ref_target[0] = 5.0 
        self.ref_target[6] = 0.5
        
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
        #self.timer = self.create_timer(self.my_params.T_s, self.control_loop)
        
        self.get_logger().info("Direct Control Node Started. Waiting for Gazebo...")

    def odom_callback(self, msg):
        """
        Convert ROS ENU Odometry to NMPC NED State
        """
        # ROS ENU (East-North-Up) -> NED (North-East-Down)
        # x_ned = y_enu
        # y_ned = x_enu
        # z_ned = z_enu
        y = msg.pose.pose.position.y
        x = msg.pose.pose.position.x
        z = msg.pose.pose.position.z

        # Orientation
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
        
        self.ekf.predict(self.u_previous)

        self.ekf.measurement_update(self.state_now)

        x_est = self.ekf.get_state_estimate()
        d_est = self.ekf.get_disturbance_estimate()


        print("state now:", self.state_now)

        print("state estimated:", x_est)


        u_optimal = self.solver.solve(x_est, self.ref_target, disturbance = d_est)
        # for i in range(8):
        #     if (u_optimal[i] - self.u_previous[i]) > 3.0:
        #         u_optimal[i] = self.u_previous[i] + 3.0
        #     elif (u_optimal[i] - self.u_previous[i]) < -3.0:
        #         u_optimal[i] = self.u_previous[i] - 3.0
                

        print("u_optimal: ", u_optimal)

        for i in range(8):
            msg = Float64()
            val = float(u_optimal[i])
            msg.data = max(-35.0, min(35.0,val))
            self.thruster_pubs[i].publish(msg)

        self.u_previous = u_optimal


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
#         for i in range(8):
#             force_msg = Float64()
            
#             # Safety
#             force_val = float(u_gazebo[i])
#             force_val = max(-35.0, min(35.0, force_val))
#             force_msg.data = force_val
            
#             self.thruster_pubs[i].publish(force_msg)

#         self.u_previous = u_optimal


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