import rclpy
from rclpy.node import Node
import time

from mavros_msgs.msg import OverrideRCIn, State
from mavros_msgs.srv import CommandBool, SetMode

class HardwareTestNode(Node):
    def __init__(self):
        super().__init__('hardware_test_node')
        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, 10)
        self.current_state = None

    def state_cb(self, msg):
        self.current_state = msg

    def wait_for_mode(self, target_mode, timeout=5.0):
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_state and self.current_state.mode == target_mode:
                return True
        return False

    def arm_boat(self):
        self.get_logger().info("Waiting for MAVROS...")
        self.arm_client.wait_for_service()
        self.mode_client.wait_for_service()

        self.get_logger().info("Setting mode to MANUAL...")
        req_mode = SetMode.Request()
        req_mode.custom_mode = "MANUAL"
        future_mode = self.mode_client.call_async(req_mode)
        rclpy.spin_until_future_complete(self, future_mode)

        if not future_mode.result() or not future_mode.result().mode_sent:
            self.get_logger().error("Failed to set MANUAL mode")
            return False

        # Wait for mode to take effect
        if not self.wait_for_mode("MANUAL"):
            self.get_logger().error("Mode did not switch to MANUAL")
            return False

        self.get_logger().info("ARMING vehicle...")
        req_arm = CommandBool.Request()
        req_arm.value = True
        future_arm = self.arm_client.call_async(req_arm)
        rclpy.spin_until_future_complete(self, future_arm)

        if future_arm.result() and future_arm.result().success:
            self.get_logger().info("Armed successfully")
            return True
        else:
            self.get_logger().error("Arming failed")
            return False
        
    def disarm_boat(self):
        """Safely disarms the vehicle."""
        self.get_logger().info("DISARMING vehicle...")
        req_arm = CommandBool.Request()
        req_arm.value = False
        future_arm = self.arm_client.call_async(req_arm)
        rclpy.spin_until_future_complete(self, future_arm)

    def send_thrust(self, throttle_pwm, steering_pwm):
        """Builds and publishes the 18-channel RC override array."""
        msg = OverrideRCIn()
        channels = [65535] * 18
        channels[0] = steering_pwm  # Channel 1: Steering
        channels[2] = throttle_pwm  # Channel 3: Throttle
        msg.channels = channels
        self.rc_pub.publish(msg)

    def run_timed_maneuver(self, duration, throttle, steering):
        """Loops the thrust command to satisfy the 1-second safety timeout."""
        start_time = time.time()
        while (time.time() - start_time) < duration:
            self.send_thrust(throttle, steering)
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    node = HardwareTestNode()
    
    # ==========================================
    # CHANGE THIS NUMBER TO TEST DIFFERENT MOVES
    # 1: Gentle Motor Test (2 seconds)
    # 2: Move Straight (5 seconds)
    # 3: Zig-Zag Right then Left (6 seconds)
    # ==========================================
    TEST_FLAG = 1
    
    # Securely arm the physical hardware
    node.arm_boat()
    
    # Execute the requested maneuver
    if TEST_FLAG == 1:
        node.get_logger().info("FLAG 1: Testing motors (2-second gentle pulse)...")
        # 1550 is very slow forward, 1500 is neutral steering
        node.run_timed_maneuver(2.0, 1550, 1500)
        
    elif TEST_FLAG == 2:
        node.get_logger().info("FLAG 2: Moving straight (5 seconds)...")
        # 1600 is moderate forward
        node.run_timed_maneuver(5.0, 1600, 1500)
        
    elif TEST_FLAG == 3:
        node.get_logger().info("FLAG 3: Moving forward and steering RIGHT (3 seconds)...")
        # 1600 throttle, 1600 steering (turns right)
        node.run_timed_maneuver(3.0, 1600, 1600)
        
        node.get_logger().info("FLAG 3: Moving forward and steering LEFT (3 seconds)...")
        # 1600 throttle, 1400 steering (turns left)
        node.run_timed_maneuver(3.0, 1600, 1400)
        
    # Always end with a safe stop and disarm
    node.get_logger().info("Stopping motors...")
    node.run_timed_maneuver(1.0, 1500, 1500)
    
    node.disarm_boat()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()