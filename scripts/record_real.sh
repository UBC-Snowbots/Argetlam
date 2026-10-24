#!/usr/bin/env bash
# Raw recorder for a real dev_arm session. Records /arm/feedback /arm/command /joint_states into an mcap bag plus a
# meta.json (git sha, arm profile, the Stepper table + its version, firmware notes, host clock info).
#
#   source /opt/ros/jazzy/setup.bash && source .../RoverFlake2/install/setup.bash
#   scripts/record_real.sh [label] [duration_s]      # duration omitted -> record until Ctrl-C
#   scripts/record_real.sh --self-test               # prove a one-shot /arm/command publish is captured (domain 42)
#
# READ-ONLY on the arm: this only subscribes (bag + a type-pin). It PUBLISHES NOTHING on the default domain, so it is
# safe to run against the live link. The type-pin (record_cmd_pin.py) + --include-unpublished-topics make the recorder
# subscribe to /arm/command at start even if no publisher exists yet -- without it the bag silently drops /arm/command
# (observed 2026-10-23: a one-shot home publish was missed, 0 command messages).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TOPICS=(/arm/feedback /arm/command /joint_states)

# rosbag2 flushes its cache and exits cleanly on SIGINT/SIGTERM; on this box `timeout`-wrapping is the reliable stop
# (a bare `kill -INT` can be ignored mid-startup). So a bounded recording runs under `timeout -s INT <dur>`.
start_pin() { python3 "$HERE/record_cmd_pin.py" & PIN=$!; }
stop_pin() { [ -n "${PIN:-}" ] && { kill -INT "$PIN" 2>/dev/null || true; wait "$PIN" 2>/dev/null || true; }; PIN=""; }

self_test() {
    export ROS_DOMAIN_ID=42   # isolated from the live arm on the default domain
    local tmp bag info cnt
    tmp="$(mktemp -d)"; bag="$tmp/bag"
    echo "self-test on ROS_DOMAIN_ID=42 (no live arm here); recording to $bag"
    start_pin
    # bag auto-stops after 8 s via SIGINT (clean flush); the one-shot publish lands inside the window.
    timeout -s INT 8 ros2 bag record -s mcap --include-unpublished-topics --topics "${TOPICS[@]}" -o "$bag" & BAG_PID=$!
    sleep 3   # let discovery resolve /arm/command's type via the pin before publishing
    echo "publishing one /arm/command (cmd_type 86 'V'); -w 1 waits for the recorder's subscription first"
    ros2 topic pub --once -w 1 /arm/command rover_msgs/msg/ArmCommand \
        "{cmd_type: 86, velocities: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], end_effector: 0.0}" >/dev/null 2>&1 || true
    wait "$BAG_PID" 2>/dev/null || true
    stop_pin
    info="$(ros2 bag info "$bag" 2>/dev/null || true)"
    cnt="$(echo "$info" | grep "/arm/command" | grep -oE "Count: [0-9]+" | grep -oE "[0-9]+" | head -1 || true)"
    echo "---- ros2 bag info ----"; echo "$info" | grep -E "Topic information|Count:|Messages:" || echo "$info" | head
    rm -rf "$tmp"
    if [ "${cnt:-0}" -ge 1 ] 2>/dev/null; then
        echo "SELF-TEST PASS: /arm/command captured ($cnt message(s)) despite no publisher at record start"
        return 0
    fi
    echo "SELF-TEST FAIL: /arm/command count = ${cnt:-0} (expected >= 1)"
    return 1
}

if [ "${1:-}" = "--self-test" ]; then self_test; exit $?; fi

LABEL="${1:-session}"
DUR="${2:-}"
PROFILE="${ROVER_ARM:-dev_arm_real}"
TS="$(date +%Y%m%d_%H%M%S)"
SESS="$REPO/data/real_sessions/${TS}_${LABEL}"
mkdir -p "$SESS"
GIT_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"

# meta.json -- stepper table comes straight from the single source of truth (control/stepper_map) so it can never
# drift from what the converter uses; keep RAW fw degrees in the bag and re-convert at read time if the table changes.
PYTHONPATH="$REPO/src" python3 - "$SESS/meta.json" "$LABEL" "$PROFILE" "$GIT_SHA" <<'PY'
import json, sys, time
from rover_intent.control.stepper_map import DEFAULT_TABLE
out, label, profile, sha = sys.argv[1:5]
meta = {
    "label": label,
    "arm_profile": profile,
    "git_sha": sha,
    "topics": ["/arm/feedback", "/arm/command", "/joint_states"],
    "stepper_table": DEFAULT_TABLE.to_dict(),
    "stepper_table_version": DEFAULT_TABLE.version,
    "firmware_notes": (
        "/arm/feedback = ArmCommand.positions[7] RAW firmware degrees (index 6 = EE), ~100 Hz, open-loop step "
        "counts (not measured encoder angle). /joint_states uses ZERO offsets in the current driver build (active "
        "ArmConstants::axis_zero_rads all 0) -> WRONG for our URDF; ignore it and convert /arm/feedback with "
        "stepper_table. /arm/command: cmd_type 86 'V' = velocities (fw deg/s), 72 'H' = home (cmd_value 170 = all "
        "but EE). Firmware travel deg [0,180/180/140/180/180/9999]; max speed deg/s 80/40/60/110/90/90. Axis 4 gear "
        "ratio is known WRONG (Rowan recalculating) and the offset table may change -> re-convert from RAW fw later. "
        "Pre-homing creep ~-0.005..-0.007 deg/s at zero command; after homing the feedback is dead still (0 deg/s)."
    ),
    "clock": {
        "note": "One clock: VM receive time = time.time() on the VM. rosbag2 stamps messages with the ROS system "
                "clock (CLOCK_REALTIME) = the same VM wall clock, so bag time and the app's recv time share an epoch.",
        "host_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host_unix": time.time(),
        "tz": time.strftime("%Z"),
    },
}
with open(out, "w") as f:
    json.dump(meta, f, indent=2)
print("wrote", out)
PY

echo "recording $LABEL -> $SESS/bag (topics: ${TOPICS[*]})"
start_pin
if [ -n "$DUR" ]; then
    echo "recording for ${DUR}s (Ctrl-C to stop early)"
    timeout -s INT "$DUR" ros2 bag record -s mcap --include-unpublished-topics --topics "${TOPICS[@]}" \
        -o "$SESS/bag" & BAG_PID=$!
    trap 'kill -INT "$BAG_PID" 2>/dev/null || true' INT TERM
    wait "$BAG_PID" 2>/dev/null || true
else
    echo "recording until Ctrl-C"
    ros2 bag record -s mcap --include-unpublished-topics --topics "${TOPICS[@]}" -o "$SESS/bag" & BAG_PID=$!
    trap 'kill -INT "$BAG_PID" 2>/dev/null || true' INT TERM
    wait "$BAG_PID" 2>/dev/null || true
fi
stop_pin
echo "done: $SESS"
