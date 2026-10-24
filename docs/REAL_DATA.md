# Real dev_arm data: record, mirror, convert, bench

How to capture data from the PHYSICAL dev_arm, mirror it into the sim, convert bags to CSV, and run the bench
characterization scripts. The live link is: laptop serial bridge -> SSH tunnel -> VM 127.0.0.1:47140 -> socat ->
/tmp/rover_arm_pty -> Rowan's stepper_arm_driver (ROS 2 Jazzy, default domain).

Topics (default domain):
- `/arm/feedback` (rover_msgs/msg/ArmCommand): `positions[7]` = RAW firmware degrees, index 6 = EE, ~100 Hz.
  Open-loop: this is commanded step counts, not a measured encoder angle.
- `/joint_states` (sensor_msgs/msg/JointState): uses ZERO offsets in the current driver build (the active
  `ArmConstants::axis_zero_rads` is all 0), so it is WRONG for our URDF. We ignore it and convert `/arm/feedback`.
- `/arm/command` (rover_msgs/msg/ArmCommand): `cmd_type` 86 'V' = velocities (fw deg/s), 72 'H' = home
  (`cmd_value` 170 = 0xAA = all axes but EE).

Firmware travel (deg): `[0, 180/180/140/180/180/9999]`. Max speed (deg/s): `80/40/60/110/90/90`.
Axis 4's gear ratio is known WRONG (Rowan recalculating) and the offset table may change, so recordings keep RAW
firmware degrees and convert only at read time.

## The one clock

Everything is stamped with **VM receive time = `time.time()` on the VM**:
- The app stamps every inbound UDP message with its own poll time (`t_recv`) and records that in the
  SessionRecorder. `real_joints` samples carry a payload `t` (the real_to_sim send time) but the app uses its OWN
  receive time, so all app-side records share one clock.
- `ros2 bag` stamps each message with the **ROS system clock**, which is `CLOCK_REALTIME` (the VM wall clock), the
  same epoch as `time.time()`. So bag time and the app's receive time are directly comparable. `real_bag_to_csv.py`
  writes the bag timestamp as `t_vm`.
- The laptop's own timestamps stay inside the payloads (e.g. BodyPose.t, rtt_ms) and are used only for latency, not
  for aligning to the arm.

Not fully verified end to end: I confirmed rosbag2's default storage uses the ROS system clock, but I could only
exercise `ros2 bag` on an isolated test domain (ROS_DOMAIN_ID=42), not against a live feedback stream timed against
`time.time()`. If you need microsecond alignment, record one short session and diff a bag timestamp against the
app's recorded `t` for the same event once; they should match to within the UDP + bag write latency.

## 1. Record (read-only, safe on the live arm)

`scripts/record_real.sh [label] [duration_s]` wraps `ros2 bag record -s mcap` of the three topics into
`data/real_sessions/<YYYYmmdd_HHMMSS>_<label>/bag` plus a `meta.json` (git sha, arm profile, the stepper table and
its version straight from `control.stepper_map`, firmware notes, host clock info). `data/` is gitignored.

    source /opt/ros/jazzy/setup.bash && source ~/projects/ubc-rover-arm/RoverFlake2/install/setup.bash
    scripts/record_real.sh teleop1 120        # record 120 s; omit the number to record until Ctrl-C

It PUBLISHES NOTHING (bag + a type-pin subscriber only), so it is safe on the default/live domain.

**/arm/command capture gotcha (fixed here):** `ros2 bag record <topics>` only subscribes to a topic whose TYPE is
discoverable. If nothing has a typed endpoint on `/arm/command` when recording starts, the recorder silently records
0 command messages, and even a later one-shot `ros2 topic pub --once` is missed (observed 2026-10-23). `record_real.sh`
fixes this two ways: it starts `record_cmd_pin.py` (a subscriber that gives `/arm/command` a typed endpoint) and runs
the recorder with `--include-unpublished-topics`, so the recorder subscribes from the start. For **one-shot** commands
(like the home), publish with `-w 1` so the publisher waits for a matching subscription before sending (otherwise the
single volatile message can beat DDS discovery). The home example in `presets.py` already uses `-w 1`.

Prove it without the arm:

    scripts/record_real.sh --self-test        # runs on ROS_DOMAIN_ID=42; prints SELF-TEST PASS when the one-shot is captured

## 2. Mirror the real arm into the sim

`scripts/real_to_sim.py` subscribes `/arm/feedback`, converts with `control.stepper_map`, and sends a new UDP message
(`real_joints`: t, fw[7], q[6]) to the app at a capped 30 Hz. On start it sends a `mirror` message so the app enters
its "mirror real" state; on exit it sends `mirror off`.

    source /opt/ros/jazzy/setup.bash && source ~/projects/ubc-rover-arm/RoverFlake2/install/setup.bash
    python3 scripts/real_to_sim.py            # live app on 127.0.0.1:47100, 30 Hz

While mirroring, the app sets the sim arm's joint TARGETS straight from `q` (no IK), refuses body/voice motion with a
clear spoken reply, and shows `mirror: true` plus the age of the last real sample in status/query. If samples go
stale (> 0.5 s) the sim holds. STOP ("stop"/"reset") leaves the mirror. `real_joints` is recorded in the
SessionRecorder like any other inbound message.

**Loop safety:** the sim->real follower (`scripts/sim_to_real.py` and the live `tools/sim_to_real.py`) reads `mirror`
from the app Query and holds at zero velocity while it is true, so real -> sim -> real never becomes a drive loop.

## 3. Convert a bag to CSV

`scripts/real_bag_to_csv.py` reads a bag with rosbag2_py and writes CSV: `t_vm` (bag receive time), `fw1..fw7` (RAW
degrees), `q1..q6` (URDF radians via `control.stepper_map`), and the table version.

    source /opt/ros/jazzy/setup.bash && source ~/projects/ubc-rover-arm/RoverFlake2/install/setup.bash
    python3 scripts/real_bag_to_csv.py data/real_sessions/<ts>_<label>/bag

By default it converts with the session's own `meta.json` table. When axis 4's gear ratio or the offsets change, edit
`control.stepper_map` (the single source of truth) or pass `--meta <other>/meta.json`, and re-run on the same bag: the
RAW degrees are kept, so old recordings re-convert cleanly.

## 4. Bench characterization (WRITE ONLY here; never run against the live link)

Each bench script requires `--i-am-at-the-bench`, prints a plan and waits for Enter, caps velocity at 0.2x firmware,
checks the start pose is inside firmware travel, sends all-zero 'V' on exit/Ctrl-C/exception, and records its own bag.
`drift_check` is the exception: it publishes nothing, so it is safe on the live arm. Run them in this order:

1. **home** the arm (so the open-loop creep is gone):
   `ros2 topic pub --once -w 1 /arm/command rover_msgs/msg/ArmCommand "{cmd_type: 72, cmd_value: 170}"`
2. `python3 scripts/bench/drift_check.py --minutes 2` (read-only) -> confirms it is homed (0 deg/s) vs creeping.
3. `python3 scripts/bench/axis_sweep.py --axis 1 --i-am-at-the-bench` -> per-axis lag, max velocity, acceleration;
   use the printed accel to replace the `max_acc` placeholder in `configs/arms/dev_arm_real.yaml`.
4. `python3 scripts/bench/axis4_check.py --i-am-at-the-bench` (WITH Rowan) -> moves axis 4 +30 fw deg, you type the
   real angle, it prints the correction factor and returns to start.
5. short teleop runs (record with `record_real.sh` while driving).
6. `python3 scripts/bench/end_check.py --i-am-at-the-bench` -> drives to the base preset with a capped P-loop, then
   you compare against the physical base mark and the result is logged.

Steppers creep ~-0.005..-0.007 deg/s at zero command BEFORE homing; after a home, `/arm/feedback` is dead still
(measured 0.000 deg/s on all axes for 497 s). So always home before a data session.

## 4a. Measured on the bench, 2026-10-23 (all six axes)

Sweeps with `axis_sweep.py --steps 5,10,16 --hold 1.5` from the homed base pose, re-analysed from the raw bags
(`data/real_sessions/*_axis_sweep_a<N>/`) with a straight-line fit of the steady part of each step (the script's own
peak-based table is noisy; trust these):

| axis | actual / commanded speed | first motion after a command | notes |
|---|---|---|---|
| 1 | 0.49 | 0.27 s | |
| 2 | 0.45 | 0.19 s | capped at 8 deg/s (0.2 x 40) |
| 3 | 0.47 | 0.22 s | |
| 4 | 0.42 | 0.17 s | gear ratio still being rechecked by Rowan: fw degrees here |
| 5 | 0.49 | 0.31 s | |
| 6 | 0.44 | 0.18 s | |

- **The firmware runs 'V' at about HALF the commanded speed**, on every axis, both directions (consistent with the
  two-phase step-pulse generation found in the firmware review). Feedback counts = generated steps, so the motors
  really move at half speed. Firmware question for Rowan; not fixed on our side.
- **Delay, not a slow ramp:** 0.17-0.31 s from a command to the first count change, then the speed is reached almost
  at once. So "acceleration" is effectively instant; a P-loop on these axes needs a low gain (kp ~1/s) or it
  overshoots (the old end_check kp 3 and the plain +v/-v sweep left 1-3 deg errors).
- **After homing the arm does not drift** (0.000 deg/s for 28 min at base). Creep only happens before homing.
- **Return-to-base overshoot on axis 6:** Rowan's driver declares "at base" at ~179.95, the arm settles ~0.6-0.7 deg
  past (180.55-180.64), twice. Same delay effect in his return loop.
- Applied: `control/stepper_map.py` MEASURED_VEL_RATIO / MEASURED_CMD_LAG_S, and dev_arm_real `max_vel` = 0.2 x fw x
  ratio (what the real arm actually reaches at the follower's default command cap), so the sim can't outrun the arm.

## 5. First data session, exact order

    # on the VM, with the live link already up (arm-link / rover-stack)
    source /opt/ros/jazzy/setup.bash && source ~/projects/ubc-rover-arm/RoverFlake2/install/setup.bash

    # (a) sanity: record the self-test proof once (no arm needed), then a real short recording
    scripts/record_real.sh --self-test
    scripts/record_real.sh warmup 30

    # (b) home, then confirm it is still
    ros2 topic pub --once -w 1 /arm/command rover_msgs/msg/ArmCommand "{cmd_type: 72, cmd_value: 170}"
    python3 scripts/bench/drift_check.py --minutes 2

    # (c) characterize (bench; someone at the e-stop)
    python3 scripts/bench/axis_sweep.py --axis 1 --i-am-at-the-bench      # repeat per axis
    python3 scripts/bench/axis4_check.py --i-am-at-the-bench              # with Rowan

    # (d) mirror the real arm into the sim to watch it (sim on a spare profile/port, or the live app on 47100)
    python3 scripts/real_to_sim.py

    # (e) short teleop runs, each wrapped in a recording
    scripts/record_real.sh teleop1 120

    # (f) end check
    python3 scripts/bench/end_check.py --i-am-at-the-bench

    # (g) convert the bags you want
    python3 scripts/real_bag_to_csv.py data/real_sessions/<ts>_teleop1/bag

Frozen-wrist teleop is on by default for the real arm (`dev_arm_real` `teleop_orient: frozen`): position-only IK on
joints 1-3, wrist held, so the wrist never spins to chase a pose. Tool tilt stays ~7 deg on a 5 cm reach, ~12 deg at
10 cm. `axis` (free the roll) is still available in the profile.
