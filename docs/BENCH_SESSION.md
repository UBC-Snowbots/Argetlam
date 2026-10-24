# In-person real-arm bench session (dev_arm)

The exact order for a live session with the physical dev_arm, using the bench tools:
`arm-link` (the serial link + Rowan's driver), `rover-follow` (preflight + follower + recorder in one command),
`rover-stack` (the Isaac sim stack), and the recorder/converter in `scripts/`.

These tools live outside and inside this repo:
- `~/bin/arm-link`      -> `tools/arm-link`       (serial link supervisor, no auto-retry by design)
- `~/bin/rover-follow`  -> `tools/rover-follow`   (one-command follow session)
- `~/bin/rover-stack`                              (the sim stack)
- `scripts/record_real.sh`, `scripts/real_to_sim.py`, `scripts/real_bag_to_csv.py` (in this repo)

Read `docs/REAL_DATA.md` first for what the data means and the bench characterization steps. This file is only the
operational runbook.

## Command sequence

Everything below runs on the VM unless it says "laptop".

```
# 0. Power on the arm. E-stop in hand. Watch the arm on every move.

# 1. (laptop) start the serial bridge + the reverse SSH tunnel to the VM:
#    serial_bridge.py  +  ssh -R 47140:localhost:<bridge> ...
#    The laptop now auto-reconnects this tunnel (it resets about once an hour, see Quirks).

# 2. (VM) source ROS once per shell
source /opt/ros/jazzy/setup.bash && source ~/projects/ubc-rover-arm/RoverFlake2/install/setup.bash

# 3. bring the serial link up (starts socat + Rowan's driver in tmux 'armlink', fully detached)
arm-link up
arm-link status            # supervisor running? how long up? last drop? (add --live to check /joint_states flows)
#   Optional, in a spare terminal: ring a bell the moment the link drops (does NOT auto-restart):
arm-link watch

# 4. home the arm so the open-loop creep is gone (confirms the e-stop first)
arm-link home

# 5. bring the stack up. Default for real-arm work: NO Isaac (mock backend, no GPU), so Thoth/other GPU jobs keep
#    the shared GPU lock and nothing waits. The real arm follows the app the same way; the HUD (skeleton, buttons,
#    camera panel) still works; you only lose the 3D Isaac view on Moonlight and sim grasp physics.
rover-stack status
rover-stack restart --no-isaac --arm dev_arm_real
#    Want the 3D view? Check the GPU is free first (tsp; nvidia-smi), then: rover-stack restart --arm dev_arm_real
#    Isaac launches OUTSIDE the tsp queue but still takes the GPU lock (gpu-run), so it waits if a GPU job runs.

# 6. put the SIM where the real arm is (the real arm homed to 'base'); this does NOT move the real arm
rover-follow align         # moves the sim to 'base' and waits until it settles

# 7. start a follow session: preflight, then recorder + follower, both detached
rover-follow on --label teleop1
#   Preflight refuses unless: arm-link is up (feedback > 50 Hz), the app is reachable on 47100, the app is NOT in
#   mirror mode, the clutch is idle, and the sim is within 1.0 deg of the real arm. If the pose check fails it tells
#   you to run `rover-follow align` first.

# 8. DRIVE. Teleop (body / plane / arm_mirror) or voice, as usual. Watch the error:
rover-follow status        # follower/recorder alive, last follower error, max |err| last 5 s, arm-link + app state

# 9. stop cleanly: zero-velocity the follower, finalize the bag
rover-follow off           # prints the follower's "stopped: zero velocity sent" line and the bag path

# 10. convert the bag you want (RAW firmware degrees are kept; conversion uses control/stepper_map)
python3 scripts/real_bag_to_csv.py data/real_sessions/<ts>_teleop1/bag

# 11. done
arm-link down              # stop the link; then power the arm off
```

### Watching the real arm in the sim (optional, instead of the follower)

The real->sim mirror and the follower are mutually exclusive (real->sim->real would be a drive loop). `rover-follow`
enforces it both ways.

```
rover-follow mirror on     # refuses if the follower is on; re-asserts 'mirror on' every ~2 s so it survives a
                           # stack restart
rover-follow mirror off    # also sends the app a final 'mirror off'
```

## Recorder path (one place for every bag)

Every bag goes to this repo's `data/real_sessions/<YYYYmmdd_HHMMSS>_<label>/` (a `bag/` directory plus `meta.json`).
`data/` is gitignored. `record_real.sh` already writes here, and `rover-follow` always records through
`record_real.sh`, so all tools agree.

Old stray bags that were written to a different place on 2026-10-23 are NOT moved; they are at:
- `~/projects/ubc-rover-arm/data/real_sessions/20261023_181259_drift_unhomed`
- `~/projects/ubc-rover-arm/data/real_sessions/20261023_182301_drift_homed_base`

If you need those, read them from that path; new sessions will not land there.

## Known quirks

- **Hourly link drop.** The laptop's `ssh -R 47140` tunnel resets about once an hour. The laptop auto-reconnects the
  tunnel, but `arm-link` does NOT auto-retry (a flapping link used to hide real faults), so the link stays stopped
  after a drop. Bring it back with `arm-link up` once the tunnel is back. `arm-link watch` prints a line + a terminal
  bell the instant it drops, with the command to run. `arm-link status` shows how long the link has been up and the
  time of the last drop.
- **Axis 6 overshoot after homing.** Rowan's driver declares "at base" at about 179.95, and the arm settles about
  0.6 to 0.7 deg past (180.55 to 180.64). Seen twice on 2026-10-23. Expected; it is the same command-lag effect in
  his return loop.
- **Real arm runs at about half the commanded speed**, on every axis and both directions, with a 0.2 to 0.3 s lag
  from a command to first motion (bench sweeps 2026-10-23; firmware two-phase pulse generation). The follower is
  closed-loop so it still converges, just slower than the sim. `dev_arm_real` `max_vel` is already set to the speed
  the real arm actually reaches, so the sim does not outrun the arm.
- **Laptop memory.** The laptop's bridge and the body/voice clients can die if the laptop runs low on memory. After a
  VM stack restart the laptop clients reconnect automatically (watchdog in `~/rover-tools/rover.py` on the laptop).
- **Do not call `rover-follow off` within a few seconds of `on`.** If the SIGTERM lands while the follower is still
  importing, it can exit before its zero-velocity `finally` runs, and `off` will warn loudly that it did not see
  "stopped: zero velocity sent". At that point the follower has not commanded any velocity yet, so the arm is safe,
  but give it a few seconds after `on` before stopping.
- **No firmware changes.** The dev_arm Teensy firmware is the team's. The half-speed and axis-4 gear-ratio issues are
  logged for Rowan; do not change firmware until the team agrees.
