# MoveIt Servo for the dev_arm (sim / mock only)

Real-time Cartesian/pose jogging of the dev_arm through MoveIt Servo (MoveIt 2.12.4 on ROS 2 Jazzy),
driving `mock_components` hardware. No Isaac, no real arm. A standalone bridge node turns Servo's joint-trajectory
output into the firmware's `rover_msgs/ArmCommand` velocity packets, for Pranav/Rowan to run on the bench later.

## What is here

| file | what it is |
|---|---|
| `dev_arm_servo.yaml` | Servo params, ported from Rowan's old-API `rover_servo_params_dev_arm.yaml` to the Jazzy schema |
| `servo_demo.launch.py` | Brings up ros2_control (mock), controllers, robot_state_publisher, and a standalone `servo_node` |
| `twist_demo.py` | Scripted twist sequence: start servo, stream TwistStamped, then go stale; prints joint snapshots |
| `run_demo.sh` | Orchestrates the demo and records `/joint_states` |
| `servo_to_arm_command.py` | **Bench adapter** (port of Rowan's `servoCallback`): Servo JointTrajectory -> `ArmCommand` 'V'. Do NOT run against the real arm from a dev machine; for Pranav/Rowan. |

We reuse the team's `dev_arm_moveit_config_v3` (URDF xacro, SRDF, kinematics, joint limits, ros2_controllers with
`mock_components/GenericSystem` + `joint_trajectory_controller`). Only the Servo params here are ours.

## Ported from Rowan (old Humble API -> Jazzy)

- Dropped (not Jazzy params): `use_gazebo`, `planning_frame`, `ee_frame_name`, `robot_link_command_frame`,
  `num_outgoing_halt_msgs_to_publish`. In new Servo the planning/EE frames come from the robot model / SRDF group
  and the twist command frame is the `TwistStamped` header `frame_id`.
- Kept Rowan's values: `command_in_type: unitless`, `scale {linear 0.5, rotational 1.2, joint 0.5}`,
  `command_out_type trajectory_msgs/JointTrajectory`, Butterworth smoothing, `check_collisions: false`
  (dev_arm URDF link geometry overlaps at home), `joint_limit_margins: [0.02]`.
- Changed: `max_expected_latency: 0.15` (Jazzy requires `publish_period < max_expected_latency/3`, so 0.1 rejects
  Rowan's 0.034); `incoming_command_timeout: 0.1` (Rowan 1.0) for a prompt stop-on-stale; singularity thresholds
  kept finite but generous (30 / 60) since Rowan had effectively disabled them (1000 / 5000).

## Build the overlay workspace (once)

`dev_arm_moveit_config_v3` and `dev_arm_description_v2` live in the read-only team repo and are not installed, so
build them into a small overlay (symlink-install, nothing copied into the team repo):

```bash
mkdir -p ~/dev_arm_servo_ws/src && cd ~/dev_arm_servo_ws/src
ln -s ~/projects/ubc-rover-arm/RoverFlake2/src/dev_arm_description_v2 .
ln -s ~/projects/ubc-rover-arm/RoverFlake2/src/dev_arm_moveit_config_v3 .
cd ~/dev_arm_servo_ws
source /opt/ros/jazzy/setup.bash
OMP_NUM_THREADS=4 colcon build --symlink-install --parallel-workers 2
```

## Run the sim/mock demo

```bash
OVERLAY_WS=~/dev_arm_servo_ws/install OUT=/tmp/servo_demo_out ROS_DOMAIN_ID=42 \
  ~/projects/ubc-rover-arm/rover-intent/ros/servo/run_demo.sh
```

Use a non-default `ROS_DOMAIN_ID` (42) so nothing can reach a real driver on domain 0. The script launches the
stack, waits for `/servo_node/start_servo`, records `/joint_states` to a rosbag in `$OUT`, runs the scripted twist
sequence (`twist_demo.py`), and prints joint snapshots showing motion then halt-on-stale.

Manual jogging instead of the script (after the launch is up). MoveIt 2.12 Servo has no `start_servo`: it starts
unpaused and you select the command type first.

```bash
ros2 service call /servo_node/switch_command_type moveit_msgs/srv/ServoCommandType "{command_type: 1}"  # 1=TWIST
ros2 topic pub /servo_node/delta_twist_cmds geometry_msgs/msg/TwistStamped \
  '{header: {frame_id: base_link}, twist: {linear: {z: 1.0}}}' -r 50
# POSE: switch_command_type command_type 2, then publish PoseStamped on /servo_node/pose_target_cmds
# pause/unpause: ros2 service call /servo_node/pause_servo std_srvs/srv/SetBool "{data: false}"
```

## Bench adapter: Servo -> real firmware (FOR PRANAV AND ROWAN ONLY)

`servo_to_arm_command.py` is the port of Rowan's `servoCallback`. It subscribes to Servo's
`/arm_controller/joint_trajectory`, converts `points[0].velocities` (rad/s) to firmware deg/s using
`fw_degps = degrees(rad/s) * dir`, with `dir = {1,1,1,1,-1,-1}` (Stepper `AXIS_DIR`), scales by `--speed`
(default 0.2), clamps to the firmware caps `{80,40,60,110,90,90}` deg/s, publishes `rover_msgs/ArmCommand`
`cmd_type='V'` on `/arm/command`, sends an all-zero 'V' after `--timeout` (0.2 s) of silence, and zeros on shutdown.

**Do not run this from a development box; it is written to be run by Pranav/Rowan on the real bench.** Exact command
(real arm attached, arm driver and bridge up, matching `ROS_DOMAIN_ID`):

```bash
# FOR PRANAV AND ROWAN, ON THE BENCH:
source /opt/ros/jazzy/setup.bash
source ~/projects/ubc-rover-arm/RoverFlake2/install/setup.bash   # rover_msgs
source ~/dev_arm_servo_ws/install/setup.bash                     # dev_arm_moveit_config_v3
# terminal A: the mock-free servo stack against the REAL driver is out of scope here; run servo + your real
#   ros2_control hardware per the team's bringup, then:
python3 ~/projects/ubc-rover-arm/rover-intent/ros/servo/servo_to_arm_command.py \
  --ros-args -p speed:=0.2 -p timeout:=0.2
```

Start slow (`speed:=0.2`), keep a hardware e-stop in reach, and confirm the firmware is in velocity ('V') mode.
