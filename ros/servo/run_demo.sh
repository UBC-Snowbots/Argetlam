#!/bin/bash
# Run the MoveIt Servo dev_arm demo on the MOCK hardware (no Isaac, no real arm) and record /joint_states to show
# motion under a scripted twist sequence, then halt-on-stale. Non-default ROS_DOMAIN_ID so nothing reaches a real
# driver on the default domain.
#
#   OVERLAY_WS=<colcon ws with dev_arm_description_v2 + dev_arm_moveit_config_v3> OUT=<dir> ros/servo/run_demo.sh
#
# See README.md for how to build OVERLAY_WS. Defaults assume the scratch build this agent made.
HERE="$(cd "$(dirname "$0")" && pwd)"
OVERLAY_WS="${OVERLAY_WS:-$HERE/../../_rosbuild/install}"
OUT="${OUT:-/tmp/servo_demo_out}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# Dev-box workaround: force UDP-only Fast-DDS so a polluted /dev/shm (stale fastrtps_* segments from other
# processes) can't break discovery. Harmless on a clean machine; unset FASTRTPS_DEFAULT_PROFILES_FILE to skip.
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$HERE/fastdds_udp_only.xml}"
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
mkdir -p "$OUT"

source /opt/ros/jazzy/setup.bash
source "$OVERLAY_WS/setup.bash"

echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID  OVERLAY_WS=$OVERLAY_WS  OUT=$OUT"
# Start the launch in its OWN process group (setsid) so teardown can kill the whole group. Otherwise some child
# nodes (servo_node, robot_state_publisher) survive a plain kill of ros2 launch, orphan on the ROS domain, and
# slow DDS discovery for the next run (which makes start_servo time out).
setsid ros2 launch "$HERE/servo_demo.launch.py" > "$OUT/servo_launch.log" 2>&1 &
LAUNCH=$!
PGID=$(ps -o pgid= -p "$LAUNCH" | tr -d ' ')
trap 'kill -INT -"$PGID" 2>/dev/null; for _ in 1 2 3 4 5 6; do kill -0 -"$PGID" 2>/dev/null || break; sleep 1; done; kill -9 -"$PGID" 2>/dev/null' EXIT

# wait for the servo node's command-type service (up to 60 s)
for i in $(seq 1 60); do
  if ros2 service list 2>/dev/null | grep -q "/servo_node/switch_command_type"; then echo "servo_node up after ${i}s"; break; fi
  sleep 1
done
ros2 control list_controllers 2>/dev/null | tee "$OUT/controllers.txt"
echo "--- topics ---"; ros2 topic list 2>/dev/null | grep -E "servo_node|joint_states|joint_trajectory" | tee "$OUT/topics.txt"

# record /joint_states for the artifact
ros2 bag record -o "$OUT/joint_states_bag" /joint_states > "$OUT/bag.log" 2>&1 &
BAG=$!
sleep 1

python3 "$HERE/twist_demo.py" 2>&1 | tee "$OUT/twist_demo.log"

kill $BAG 2>/dev/null; sleep 1
echo "=== servo_node log (tail) ==="; grep -iE "servo|halt|singular|limit|error|warn" "$OUT/servo_launch.log" | tail -20
