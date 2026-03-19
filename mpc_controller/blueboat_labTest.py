#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time

from mavros_msgs.msg import OverrideRCIn, State
from mavros_msgs.srv import CommandBool, SetMode

class GentleTestNode(Node):
    def __init__(self):
        super().__init__('gentle_test_node')
        self.rc_pub = self.create_publisher(OverrideRCIn, '/mavros/rc/override', 10)
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, 10)
        self.current_state = None
        self.state_received = False

    def state_cb(self, msg):
        self.current_state = msg
        self.state_received = True

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
        # Wait for state topic
        while not self.state_received and rclpy.ok():
            self.get_logger().info('Waiting for state topic...')
            rclpy.spin_once(self, timeout_sec=0.5)

        # Set mode to MANUAL
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

        # --- Multiple arming attempts (community fix) ---
        self.get_logger().info('Arming with multiple attempts...')
        req_arm = CommandBool.Request()
        req_arm.value = True
        armed = False

        for attempt in range(4):  # 4 attempts as suggested
            self.get_logger().info(f'Arming attempt {attempt+1}/4')
            future_arm = self.arm_client.call_async(req_arm)
            rclpy.spin_until_future_complete(self, future_arm, timeout_sec=2.0)
            if future_arm.result() and future_arm.result().success:
                self.get_logger().info(f'Attempt {attempt+1} service reported success')
            else:
                self.get_logger().warn(f'Attempt {attempt+1} service did not report success')

            # Short delay between attempts to let the FCU process
            time.sleep(0.5)

            # Check if already armed via state (optional early exit)
            if self.current_state and self.current_state.armed:
                armed = True
                break

        if not armed:
            # Final wait for armed state with timeout
            if self.wait_for_armed(True, timeout=5.0):
                armed = True

        if armed:
            self.get_logger().info('Armed successfully')
            return True
        else:
            self.get_logger().error('Vehicle did not become armed')
            return False

    def disarm_gentle(self):
        self.get_logger().info('Disarming with multiple attempts...')
        req_arm = CommandBool.Request()
        req_arm.value = False
        
        for attempt in range(4):
            self.get_logger().info(f'Disarm attempt {attempt+1}/4')
            future_arm = self.arm_client.call_async(req_arm)
            rclpy.spin_until_future_complete(self, future_arm, timeout_sec=2.0)
            
            if future_arm.result() and future_arm.result().success:
                self.get_logger().info('Disarmed successfully!')
                return True
            else:
                self.get_logger().warn(f'Attempt {attempt+1} service did not report success')
            
            time.sleep(0.5)
            
            # Check if state actually changed to disarmed
            if self.current_state and not self.current_state.armed:
                self.get_logger().info('Verified disarmed via state topic.')
                return True
                
        self.get_logger().error('CRITICAL: Vehicle refused to disarm!')
        return False

    def send_thrust(self, throttle_pwm, steering_pwm=1600):
        msg = OverrideRCIn()
        channels = [65535] * 18
        channels[0] = steering_pwm   # Steering (channel 1)
        channels[2] = throttle_pwm   # Throttle (channel 3)
        msg.channels = channels
        self.rc_pub.publish(msg)

    def release_overrides(self):
        self.get_logger().info('Releasing RC overrides to neutral...')
        msg = OverrideRCIn()
        # 65535 tells ArduPilot to ignore the override and return to its own neutral
        msg.channels = [65535] * 18 
        self.rc_pub.publish(msg)

    def run_thrust(self, duration, throttle, steering=1600):
        start = time.time()
        while time.time() - start < duration:
            self.send_thrust(throttle, steering)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)
        
        # PROPER WAY TO STOP: Release overrides instead of guessing neutral
        self.release_overrides()
        time.sleep(0.5) # Give ArduPilot a half-second to register neutral before disarming

def main(args=None):
    rclpy.init(args=args)
    node = GentleTestNode()

    # === CONFIGURATION ===
    THROTTLE_PWM = 1520        # Adjust as needed (1550–1650 for a clear test)
    DURATION = 2.0             # seconds
    # =====================

    success = node.arm_gentle()
    if success:
        node.get_logger().info(f'Sending throttle {THROTTLE_PWM} for {DURATION} sec...')
        node.run_thrust(DURATION, THROTTLE_PWM)
        node.disarm_gentle()
    else:
        node.get_logger().error('Aborting test.')

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()