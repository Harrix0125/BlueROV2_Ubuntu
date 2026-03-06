import rclpy
from rclpy.node import Node
import time

# MAVROS messages and services
from mavros_msgs.msg import OverrideRCIn
from mavros_msgs.srv import CommandBool, SetMode

class BenchTestNode(Node):
    def __init__(self):
        super().__init__('blueboat_bench_test')
        
        # Publisher for the thrusters
        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)
        
        # Service clients for Arming and Modes
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

    def set_mode(self, mode_string):
        """Changes the flight controller mode."""
        req = SetMode.Request()
        req.custom_mode = mode_string
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().mode_sent

    def arm_vehicle(self, arm_state):
        """Arms or Disarms the thrusters."""
        req = CommandBool.Request()
        req.value = arm_state
        future = self.arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success

    def send_thrust(self, throttle_pwm, steering_pwm):
        """Sends PWM signals to the motors."""
        msg = OverrideRCIn()
        channels = [65535] * 18
        
        channels[0] = steering_pwm
        channels[2] = throttle_pwm 
        
        msg.channels = channels
        self.rc_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = BenchTestNode()

    node.get_logger().info("Waiting for MAVROS connection...")
    node.arm_client.wait_for_service()
    node.mode_client.wait_for_service()

    # Step 1: Set to MANUAL mode
    node.get_logger().info("Setting mode to MANUAL...")
    node.set_mode("MANUAL")
    time.sleep(1)

    # Step 2: Arm the boat
    node.get_logger().info("ARMING thrusters! Keep hands clear.")
    node.arm_vehicle(True)
    time.sleep(1)

    # Step 3: Spin motors for 3 seconds
    # Note: ArduRover requires constant messages, so we loop for 3 seconds
    node.get_logger().info("Sending Forward Thrust (PWM 1550) for 3 seconds...")
    start_time = time.time()
    while (time.time() - start_time) < 3.0:
        node.send_thrust(1550, 1500) # 1550 is slow forward, 1500 is no steering
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.1)

    # Step 4: Stop motors
    node.get_logger().info("Stopping motors...")
    node.send_thrust(1500, 1500)
    rclpy.spin_once(node, timeout_sec=0.1)
    time.sleep(1)

    # Step 5: Disarm
    node.get_logger().info("DISARMING vehicle...")
    node.arm_vehicle(False)

    node.get_logger().info("Bench test complete!")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()