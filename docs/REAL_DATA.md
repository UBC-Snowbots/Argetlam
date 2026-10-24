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

## 6. Gripper (EE, firmware axis 7 'ee_prismatic')

The gripper is a lead-screw LINEAR axis, driven through Rowan's driver as the `end_effector` velocity (fw deg/s) in
a cmd_type 86 'V' command; feedback is `/arm/feedback` positions[6]. THREE facts make it dangerous and finicky:

- NO limit switch, and home-all (cmd_value 170 = 0xAA) skips it, so its zero is ARBITRARY after every power-on.
  Everything below is RELATIVE TO THIS POWER-ON. Re-mark (or re-reference) EACH SESSION.
- The firmware converts EE steps with AXIS 6's gear ratio (a known bug), so the reported EE "degrees" are a
  mis-scaled, arbitrary unit. Only RELATIVE positions mean anything; never read them as an angle or a length.
- One end has a mechanical HARD STOP (Pranav thinks the opening end, unconfirmed); the other can SQUEEZE/STALL.
  Treat BOTH ends as dangerous until marked. Feedback is OPEN-LOOP (commanded step counts), so a mechanical stall
  does NOT show in the numbers -- watch and LISTEN, and stop with any non-move key.

### 6a. Mark the limits at the bench (Pranav, at the arm)

    source /opt/ros/jazzy/setup.bash && source ~/projects/ubc-rover-arm/RoverFlake2/install/setup.bash
    python3 scripts/bench/ee_limits.py --i-am-at-the-bench        # prints a plan, waits for Enter

It drives ONLY the EE (joints held at 0), at <= 10 fw deg/s (hard cap 30), and records its own bag via
`record_real.sh`. First it CALIBRATES the open direction: tap `o` once (it nudges one step in +fw), then type whether
the fingers OPENED or CLOSED -> that fixes `open_sign` (which fw-velocity sign opens). Then jog and mark:

    o = open one step      c = close one step      (holding a key = OS auto-repeat = repeated steps)
    [ = halve the step     ] = double the step
    m = mark here (it then asks: open-stop or closed?)   s = save   q / Ctrl-C = stop and quit
    any other key          = STOP (zero velocity)

Jog gently to the OPEN stop, press `m`, answer `o`. Jog to the CLOSED end, press `m`, answer `c`. Press `s` to save
to `configs/arms/dev_arm_real_ee.yaml`. The saved numbers:

- `open_fw`, `closed_fw`  : EE fw deg at the two marks (mis-scaled unit, relative to this power-on).
- `open_sign`             : +1 or -1, the fw-velocity sign that OPENS the fingers.
- `margin_fw`             : fw deg kept clear of each stop (default 5). SOFT LIMITS = the [open, closed] travel
                            contracted by `margin_fw` at each end; the follower never commands outside them.
- `marked_at`, `notes`    : a reminder the reference is per-power-on.

WATCH/LISTEN FOR: grinding or a motor that keeps whining while the fingers have stopped = a stall into a stop (the
feedback keeps counting, so it will NOT tell you). Back off a step and re-mark a touch inside.

### 6b. Re-reference a later session (faster than re-marking both ends)

    python3 scripts/bench/ee_limits.py --i-am-at-the-bench --re-reference

Jog to the OPEN position by eye (o/c), press `m`, press `s`. It shifts the saved open/closed so OPEN matches where
you are now, KEEPING the span. Use this at the start of each session instead of re-marking both ends.

### 6c. Drive the gripper from the sim (follower), GATED

The sim->real follower drives the gripper ONLY when BOTH are true: the limits file exists AND `--ee` is passed (the
config equivalent is `real_arm.ee: true` in `dev_arm_real.yaml`, default false). Otherwise `end_effector` stays 0
everywhere.

    python3 tools/sim_to_real.py --axes 1,2,3,5,6 --ee        # joints + gripper
    python3 tools/sim_to_real.py --axes 1,2,3,5,6 --ee --ee-speed 8   # slower gripper

The app's gripper value (`grip` 0 open .. 1 closed) maps to an EE target `open_fw + grip*(closed_fw - open_fw)`,
clamped to the soft limits, driven with its own slow capped P-loop (default 10 fw deg/s). On start the follower
REFUSES EE control if the current EE feedback is outside `[closed - 2*margin, open + 2*margin]` (a stale per-power-on
reference -- a fresh power-on or it crept); it then runs with EE OFF and tells you to re-reference. `end_effector` is
zeroed on stale feedback / mirror / homing / exit, exactly like the joints.

The `real_to_sim` mirror shows the real EE back as the sim gripper value (inverse map) when the limits file exists,
so you can watch the real gripper state on the sim.

### 6d. Open question for Rowan

Is it OK to gently drive the gripper INTO the hard stop to get a repeatable open reference (the lead-screw would
stall there), or must we never stall it? If stalling the hard stop is safe, the re-reference step can use a light
touch of that stop instead of eyeballing the open position. Until Rowan confirms, re-reference BY EYE only.

### 6e. What is not verified until the real gripper is in front of us

Direction (which fw sign opens -> `open_sign`), the EE speed units (mis-scaled fw deg/s; start low), and the stall
behaviour at each end are all UNKNOWN until Pranav marks them at the bench. The code starts slow and gated for that
reason.
