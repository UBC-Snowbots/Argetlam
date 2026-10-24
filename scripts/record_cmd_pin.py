#!/usr/bin/env python3
"""Type-pin for /arm/command so `ros2 bag record --include-unpublished-topics` captures it from the very start.

Why this exists: `ros2 bag record -s mcap /arm/feedback /arm/command /joint_states` only subscribes to a topic
whose TYPE is discoverable in the ROS graph. If no node has a typed endpoint on /arm/command when the recorder
starts, the recorder cannot learn rover_msgs/msg/ArmCommand and silently records 0 command messages, even a later
one-shot `ros2 topic pub --once` is missed (observed 2026-10-23). In the real session the driver subscribes to
/arm/command and provides the type, but we do not want recording to depend on the driver being up at that instant.

This node just holds a SUBSCRIPTION to /arm/command (rover_msgs/msg/ArmCommand). A subscription is a typed graph
endpoint, so `--include-unpublished-topics` can resolve the type and subscribe at start. It PUBLISHES NOTHING, so
it is safe on the default (live) ROS domain. It counts what it sees and prints a line on exit; it never commands.

    source /opt/ros/jazzy/setup.bash && source .../RoverFlake2/install/setup.bash
    python3 record_cmd_pin.py            # Ctrl-C to stop
"""
import sys

import rclpy
from rclpy.node import Node
from rover_msgs.msg import ArmCommand


class CmdPin(Node):
    def __init__(self):
        super().__init__("record_cmd_pin")
        self.n = 0
        self.create_subscription(ArmCommand, "/arm/command", self._on, 10)

    def _on(self, _msg):
        self.n += 1


def main():
    rclpy.init()
    node = CmdPin()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stderr.write(f"record_cmd_pin: saw {node.n} /arm/command message(s)\n")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
