"""MoveIt Servo for the dev_arm, mock hardware only (no Isaac, no real arm).

Brings up: ros2_control_node with the team's mock_components system, joint_state_broadcaster + arm_controller,
robot_state_publisher, a world->base_link static TF, and a standalone moveit_servo servo_node. The servo_node
accepts TwistStamped on /servo_node/delta_twist_cmds and PoseStamped on /servo_node/pose_target_cmds, respects the
URDF joint limits, and halts on stale commands (incoming_command_timeout). Select the command type first with the
/servo_node/switch_command_type service (moveit_msgs/ServoCommandType: 0 JOINT_JOG, 1 TWIST, 2 POSE), then publish
commands. (MoveIt 2.12 Servo has no start_servo; /servo_node/pause_servo SetBool pauses/unpauses.)

Reuses the team's dev_arm_moveit_config_v3 (URDF xacro, SRDF, kinematics, joint_limits, ros2_controllers) via an
overlay workspace; only the Servo params (dev_arm_servo.yaml, next to this file) are ours. Run on a non-default
ROS_DOMAIN_ID (e.g. 42) so nothing can reach a real driver on the default domain.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

HERE = os.path.dirname(os.path.abspath(__file__))
MOVEIT_PKG = "dev_arm_moveit_config_v3"   # team package, provided by the overlay workspace (see README)


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("dev_arm", package_name=MOVEIT_PKG)
        .robot_description(file_path="config/dev_arm.urdf.xacro")
        .robot_description_semantic(file_path="config/dev_arm.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .to_moveit_configs()
    )

    with open(os.path.join(HERE, "dev_arm_servo.yaml")) as f:
        servo_params = {"moveit_servo": yaml.safe_load(f)}

    ros2_controllers = os.path.join(get_package_share_directory(MOVEIT_PKG), "config", "ros2_controllers.yaml")

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers],
        remappings=[("/controller_manager/robot_description", "/robot_description")],
        output="screen",
    )
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_world_base",
        arguments=["--frame-id", "world", "--child-frame-id", "base_link"],
    )
    jsb_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager", "--controller-manager-timeout", "60"],
    )
    arm_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["arm_controller", "-c", "/controller_manager", "--controller-manager-timeout", "60"],
    )
    servo_node = Node(
        package="moveit_servo",
        executable="servo_node",
        name="servo_node",
        output="screen",
        parameters=[
            servo_params,
            {"update_period": 0.034},          # acceleration-limiting filter period
            {"planning_group_name": "arm"},
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
    )
    return LaunchDescription([ros2_control_node, rsp, static_tf, jsb_spawner, arm_spawner, servo_node])
