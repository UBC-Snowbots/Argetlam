"""Bridge: MoveIt Servo joint-trajectory output -> rover_msgs/ArmCommand 'V' (velocity) for the dev_arm firmware.

A standalone port of Rowan's commented-out servoCallback (RoverFlake2 arm_control/moveit_control.h). Servo publishes
a trajectory_msgs/JointTrajectory on /arm_controller/joint_trajectory; this node reads points[0].velocities (rad/s,
in the 6 arm-joint order), converts to firmware deg/s, and publishes rover_msgs/ArmCommand{cmd_type='V'} on
/arm/command.

Conversion (firmware frame): the firmware/URDF relation is q_rad = radians(fw_deg) * dir + zero, so a joint velocity
maps as  fw_degps = degrees(rad_per_s) * dir  (the fixed `zero` offset drops out for velocity). dir is Rowan's
Stepper table AXIS_DIR {1, 1, 1, 1, -1, -1}. The result is scaled by --speed (default 0.2, matching the sim_to_real
follower) and hard-clamped to the firmware per-axis caps {80, 40, 60, 110, 90, 90} deg/s. If no Servo command
arrives for --timeout seconds (default 0.2) the node sends an all-zero 'V'; it also sends one on shutdown.

!!! DO NOT RUN against the real arm from this agent. This file is written for Pranav/Rowan to run on the bench;
    see README.md. The pure mapping/timeout logic below is unit-tested with a fake publisher (no ROS needed).
"""
from __future__ import annotations

import math
from collections import namedtuple

CMD_V = ord("V")                       # 86: ArmCommand.cmd_type for a velocity command
AXIS_DIR = (1, 1, 1, 1, -1, -1)        # Rowan's Stepper table directions (joy_arm_control.h AXIS_DIR)
FIRMWARE_CAPS_DEGPS = (80.0, 40.0, 60.0, 110.0, 90.0, 90.0)  # FirmFlake main.cpp per-axis max speeds
N = 6

# What the ROS node publishes, kept ROS-free so the logic is testable with a fake publisher.
ArmCmd = namedtuple("ArmCmd", ["cmd_type", "velocities"])


class ServoToArmMapper:
    """Pure logic: velocity mapping + stale-command timeout. No ROS imports, so it unit-tests directly."""

    def __init__(self, speed: float = 0.2, timeout: float = 0.2,
                 dir_: tuple = AXIS_DIR, caps: tuple = FIRMWARE_CAPS_DEGPS):
        self.speed = float(speed)
        self.timeout = float(timeout)
        self.dir = tuple(dir_)
        self.caps = tuple(caps)
        self._last_cmd_t: float | None = None
        self._stopped = True            # true once a zero has been sent (don't re-spam zeros)

    def map_velocities(self, vel_rad) -> list[float]:
        """rad/s (6, Servo/URDF frame) -> firmware deg/s (scaled by speed, clamped to caps, direction-corrected)."""
        if len(vel_rad) != N:
            raise ValueError(f"expected {N} joint velocities, got {len(vel_rad)}")
        out = []
        for i in range(N):
            v = math.degrees(float(vel_rad[i])) * self.dir[i] * self.speed
            out.append(max(-self.caps[i], min(self.caps[i], v)))
        return out

    def zero_command(self) -> ArmCmd:
        return ArmCmd(CMD_V, [0.0] * N)

    def on_trajectory(self, point_velocities_rad, now: float) -> ArmCmd:
        """A fresh Servo trajectory point arrived: record the time and return the 'V' command to publish."""
        self._last_cmd_t = now
        self._stopped = False
        return ArmCmd(CMD_V, self.map_velocities(point_velocities_rad))

    def poll(self, now: float) -> ArmCmd | None:
        """Call periodically. Returns an all-zero 'V' once when commands have gone stale (> timeout), else None."""
        if self._last_cmd_t is None or self._stopped:
            return None
        if now - self._last_cmd_t > self.timeout:
            self._stopped = True
            return self.zero_command()
        return None


# ---------------------------------------------------------------------------------------------------------------
# ROS node (imported lazily so this module stays importable without a ROS environment).
def main():
    import rclpy
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectory
    from rover_msgs.msg import ArmCommand

    class ServoToArmCommand(Node):
        def __init__(self):
            super().__init__("servo_to_arm_command")
            self.declare_parameter("speed", 0.2)
            self.declare_parameter("timeout", 0.2)
            self.declare_parameter("in_topic", "/arm_controller/joint_trajectory")
            self.declare_parameter("out_topic", "/arm/command")
            self.m = ServoToArmMapper(speed=self.get_parameter("speed").value,
                                      timeout=self.get_parameter("timeout").value)
            self.pub = self.create_publisher(ArmCommand, self.get_parameter("out_topic").value, 10)
            self.create_subscription(JointTrajectory, self.get_parameter("in_topic").value, self._on_traj, 10)
            self.create_timer(self.m.timeout / 2.0, self._tick)
            self.get_logger().info(f"servo_to_arm_command: speed={self.m.speed} timeout={self.m.timeout}s "
                                   f"caps={self.m.caps} dir={self.m.dir}")

        def _now(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def _publish(self, cmd: ArmCmd):
            msg = ArmCommand()
            msg.cmd_type = cmd.cmd_type
            msg.velocities = list(cmd.velocities)
            self.pub.publish(msg)

        def _on_traj(self, msg: JointTrajectory):
            if not msg.points or len(msg.points[0].velocities) != N:
                self.get_logger().warn("ignoring trajectory: no points / wrong joint count")
                return
            self._publish(self.m.on_trajectory(list(msg.points[0].velocities), self._now()))

        def _tick(self):
            stale = self.m.poll(self._now())
            if stale is not None:
                self.get_logger().info("commands stale -> sending zero velocity")
                self._publish(stale)

        def stop(self):
            self._publish(self.m.zero_command())

    rclpy.init()
    node = ServoToArmCommand()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()                      # zero on shutdown
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
