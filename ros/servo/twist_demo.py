"""Scripted twist sequence for the servo demo (sim/mock only).

MoveIt 2.12 / Jazzy Servo has no start_servo service. Instead it starts unpaused and ignores a command stream until
you select the command type: this unpauses via /servo_node/pause_servo (SetBool false) and selects TWIST via
/servo_node/switch_command_type (moveit_msgs/ServoCommandType), then streams TwistStamped to
/servo_node/delta_twist_cmds for a few seconds, then STOPS sending so the servo's incoming_command_timeout halts the
arm (stop-on-stale). Then it switches to POSE and sends one PoseStamped to show pose commands are accepted. Prints
/joint_states before/during/after, so the caller sees motion then halt. Pure ROS client; never touches a real driver.

  python3 twist_demo.py            # +Z then +X twist, then stale, then a pose command
"""
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from moveit_msgs.srv import ServoCommandType
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

ARM_JOINTS = ["shoulder_joint", "link_1_joint", "link1_link2", "a4_rotation", "a5_rotation", "a6_rotation"]


class TwistDemo(Node):
    def __init__(self):
        super().__init__("servo_twist_demo")
        self.twist_pub = self.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/servo_node/pose_target_cmds", 10)
        self.js = {}
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.switch_cli = self.create_client(ServoCommandType, "/servo_node/switch_command_type")
        self.pause_cli = self.create_client(SetBool, "/servo_node/pause_servo")

    def _on_js(self, msg):
        self.js = dict(zip(msg.name, msg.position))

    def arm_snapshot(self):
        return [round(self.js.get(j, float("nan")), 4) for j in ARM_JOINTS]

    def spin_for(self, secs):
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def _call(self, cli, req, tries=5):
        for k in range(tries):
            self.spin_for(0.5)                       # let discovery settle / robot state arrive
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            if fut.result() and getattr(fut.result(), "success", True):
                return True, k + 1
        return False, tries

    def enable_twist(self):
        self.get_logger().info("waiting for /servo_node/switch_command_type ...")
        if not self.switch_cli.wait_for_service(timeout_sec=30.0):
            self.get_logger().error("switch_command_type service never appeared")
            return False
        if self.pause_cli.wait_for_service(timeout_sec=5.0):   # unpause (SetBool data=false)
            self._call(self.pause_cli, SetBool.Request(data=False))
        ok, n = self._call(self.switch_cli, ServoCommandType.Request(command_type=ServoCommandType.Request.TWIST))
        self.get_logger().info(f"switch to TWIST -> {ok} (attempt {n})")
        return ok

    def set_pose_mode(self):
        ok, n = self._call(self.switch_cli, ServoCommandType.Request(command_type=ServoCommandType.Request.POSE))
        self.get_logger().info(f"switch to POSE -> {ok} (attempt {n})")
        return ok

    def stream_twist(self, lin=(0.0, 0.0, 1.0), ang=(0.0, 0.0, 0.0), secs=2.0, hz=50.0, frame="base_link"):
        period, end, samples = 1.0 / hz, time.time() + secs, []
        while time.time() < end:
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = frame
            m.twist.linear.x, m.twist.linear.y, m.twist.linear.z = lin
            m.twist.angular.x, m.twist.angular.y, m.twist.angular.z = ang
            self.twist_pub.publish(m)
            self.spin_for(period)
            samples.append((time.time(), self.arm_snapshot()))
        return samples

    def sample(self, secs, hz=20.0):
        """Watch /joint_states for `secs` seconds without sending anything."""
        period, end, samples = 1.0 / hz, time.time() + secs, []
        while time.time() < end:
            self.spin_for(period)
            samples.append((time.time(), self.arm_snapshot()))
        return samples

    def send_pose(self, frame="base_link"):
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = frame
        m.pose.orientation.w = 1.0
        self.pose_pub.publish(m)
        self.spin_for(0.2)


def _max_speed(samples):
    """Max per-joint angular speed (rad/s) seen across consecutive samples (ignoring NaN)."""
    v = 0.0
    for (t1, a), (t2, b) in zip(samples, samples[1:]):
        dt = t2 - t1
        if dt <= 0:
            continue
        for x, y in zip(a, b):
            if not (math.isnan(x) or math.isnan(y)):
                v = max(v, abs(x - y) / dt)
    return v


def main():
    rclpy.init()
    n = TwistDemo()
    n.spin_for(1.0)
    before = n.arm_snapshot()
    print("joints before      :", before)
    assert n.enable_twist(), "could not switch Servo to TWIST"
    act = n.stream_twist(lin=(0.0, 0.0, 1.0), secs=2.0)        # +Z up
    print("joints during (+Z) :", n.arm_snapshot())
    act += n.stream_twist(lin=(1.0, 0.0, 0.0), secs=1.5)       # +X forward
    print("joints during (+X) :", n.arm_snapshot())
    active_speed = _max_speed(act)
    # STOP sending. Allow 0.8 s for the controller to finish the last micro-trajectory, then measure the speed over
    # the next 1.5 s: it should be ~0 (servo halts on stale commands via incoming_command_timeout).
    settle = n.sample(0.8)
    stale = n.sample(1.5)
    stale_speed = _max_speed(stale)
    print("joints stale start :", settle[0][1] if settle else None)
    print("joints stale end   :", stale[-1][1] if stale else None)
    print(f"max joint speed active (rad/s): {active_speed:.4f}")
    print(f"max joint speed after settle  (rad/s): {_max_speed(settle):.4f}")
    print(f"max joint speed stale   (rad/s): {stale_speed:.4f}")
    n.set_pose_mode()
    n.send_pose()
    n.spin_for(0.5)
    print("joints after pose  :", n.arm_snapshot())
    print("MOVED_ON_TWIST:", active_speed > 0.03)
    print("HALTED_WHEN_STALE:", stale_speed < 0.01)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
