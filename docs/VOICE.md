# Voice commands

The laptop (rover-voice: push-to-talk + local faster-whisper base.en) sends each finished transcript to the VM app
as an `utterance`. The app handles it in three layers, fastest first:

1. **STOP / reset (instant, in-tick):** `nemotron.urgent_stop` matches stop/freeze/halt/hold on/wait/whoa/abort and
   reset/restart/start over locally, before anything else. STOP always wins and cancels any armed confirm, running
   preset, or running sequence.
2. **Fast local path (in-tick, < 50 ms, no cloud):** `nemotron.quick_pose` (short named poses) and
   `planner/fast_path.parse_fast` (the motion grammar below). If the WHOLE utterance is covered, it runs in the same
   control tick with no Nemotron round trip. Anything it does not fully understand (a complex sentence, an object
   name, free phrasing) returns None and falls through.
3. **Nemotron (cloud, 1.4-3.9 s):** everything else. The schema now also carries `jog` / `joint_jog` / `speed`, so
   free-phrased motion still works; the result is mapped back onto the same steps as the fast path. The cloud call
   also gets a compact CONTEXT string (the last 3 executed commands + the current named pose) so free phrasing that
   refers back ("spin around the other way", "do that again") resolves. The client has a timeout (`nemotron.timeout_s`,
   default 6 s); on a timeout the app says "Sorry, I didn't catch that in time" instead of hanging.

The live finding (2026-10-23 real mic) was that a named pose took a Nemotron round trip plus a spoken-confirm window,
so a slightly late "yes" timed out. Fix: poses and the motion commands below are parsed locally and, on the real arm,
**move at once with no spoken confirm** (see the confirm switches).

## Confirm switches (`confirm:` in config)

| key | default (`default.yaml`, sim) | `dev_arm_real.yaml` (real arm) | gates |
|-----|------|------|-------|
| `presets` | `true` (ask "say yes") | `false` (move at once) | named poses |
| `motion`  | `true` | `false` | Cartesian jog, joint jog, sequences, speed |
| `skills`  | `false` | `true` | autonomous pick / place / move (they aim at SIM object positions) |

STOP always bypasses every confirm. Named poses and jogs are bounded (joint limits + the workspace box), which is why
they are safe to run with no confirm on the real arm; skills stay gated because the real table does not match the sim.

## Named poses (quick_pose)

`stow` / `rest` -> **base** (the real rest pose), `ready` -> **ready**, `go back` / `return` / `back to start` /
`home` -> **the sim start pose** (RETURN; voice "home" NEVER homes the hardware). Short filler is allowed
("go to the ready pose", "stow the arm please"). Whisper mishears folded to poses: store/stove/stowe/stoe -> stow,
reddy/readi -> ready. A clause that names a scene object ("store the cup") is NOT a pose and goes to Nemotron.

## Motion grammar (fast_path.parse_fast)

Frame: robot BASE frame as the operator faces the arm. **+x = forward, +y = left, +z = up.**

- **Cartesian jog:** a direction in {left, right, up, down, forward, back} + an optional distance. Default 5 cm;
  "a bit" / "a little" = 3 cm; clamped to <= 15 cm. Units: cm / centimetre(s), mm / millimetre(s). The app solves IK
  for TCP + delta holding the current tool orientation (iterated to convergence), checks joint limits and the
  workspace-box path, then runs it through the same joint_goal -> SafetyLayer path as the presets. A jog it cannot
  reach within 1 cm, or that leaves the box, is refused with a spoken reason.
  - e.g. "move left 5 centimetres", "nudge up 3 cm", "go forward ten cm", "move back a bit".
  - mishears folded: write/rite/wright -> right, "for word"/"foward" -> forward.
- **Joint jog:** one joint by name + a direction word + an optional angle. Default 10 deg; "a bit" = 5 deg; clamped
  to <= 45 deg. The delta is clamped into the joint limits. Names and the MEASURED sign map (dev_arm_real at home,
  2026-10-23 - a direction word always moves the TCP the way the operator means):

  | say | joint | up/left = | down/right = |
  |-----|-------|-----------|--------------|
  | base / turn / rotate | J1 | left = +q1 (ccw from above) | right = -q1 (cw) |
  | shoulder | J2 | up = -q2 | down = +q2 |
  | elbow | J3 | up = +q3 | down = -q3 |
  | forearm / roll | J4 | ccw / in = +q4 | cw / out = -q4 (a roll; TCP barely moves) |
  | wrist / tilt | J5 | up = -q5 | down = +q5 |
  | tool / twist | J6 | ccw / in = +q6 | cw / out = -q6 (a roll) |

  - e.g. "turn the base left 30 degrees", "shoulder up", "elbow up 20 degrees", "twist the tool clockwise 45 degrees".
  - **Default joint when none is named (so you need not know the joint layout):**
    - "rotate / turn / spin [left|right|clockwise|counterclockwise] [by] N degrees" -> the **base** (J1). With no
      direction word it defaults to the base's positive direction (left / counterclockwise).
    - "left | right [by] N degrees" (no joint) -> the **base** (J1).
    - "up | down [by] N degrees" (no joint) -> the **shoulder** (J2).
    - A rotate/turn with a joint but no direction defaults to that joint's positive direction, so
      "rotate the elbow by 10 degrees" parses (elbow up 10). Degrees are required for these defaults, so a plain
      Cartesian jog ("move left 5 cm") and a bare "spin around" are untouched (the latter still goes to Nemotron).
    The same defaults are stated in the Nemotron prompt, so the fast path and the cloud agree.

- **Context ("again" / "the other way"):** the app keeps the last 5 executed motion commands (with their resolved
  parameters) and the current named pose. The fast path resolves, with no cloud call:
  - "again" / "repeat" / "once more" / "do it again" -> redo the last motion command.
  - "the other way" / "opposite" / "reverse" / "undo" / "back the other way" -> the last motion REVERSED: a Cartesian
    jog flips direction, a joint jog negates the angle, and a preset goes back to the previous pose (if known).
  - With no history yet (or a reverse we can't form) the whole utterance defers to Nemotron, which gets the same
    context string so free phrasing like "spin around the other way" still works.
  - STOP clears nothing here: a new command after STOP still sees the history.

- **Grammar robustness:** US and UK spellings and plurals (centimeter(s)/centimetre(s)/cm, millimeter(s)/mm,
  meter(s)/metre(s)/m, degree(s)/deg); number words up to ninety incl. compounds ("forty five") and halves
  ("half a cm", "one and a half degrees"); "by N" and bare "N"; and polite fillers / trailing punctuation
  ("please", "can you", "could you", "now", "and", ".") are stripped. Every valid motion command from the
  2026-10-23 real-mic session now parses on the fast path (see tests/test_voice_fast.py).
- **Sequences:** "X then Y then Z" / "X and then Y" (also comma-separated), up to 5 steps. Each step starts when the
  previous one reaches its goal (`at_goal`). STOP, any new command, or the clutch cancels the rest. A step that is
  refused stops the sequence.
  - e.g. "move left 5 cm then up 5 cm then go to ready", "stow then home".
- **Speed:** "slower" / "faster" / "normal speed" scales the SafetyLayer velocity cap for voice moves between 0.25x
  and 1.0x of the profile (never above the profile's own limits). slower halves, faster doubles, normal resets.

## Not the fast path

Anything naming a scene object ("pick up the cup", "put the cup left of the bottle"), "follow me", calibration, and
free phrasing go to Nemotron (or the offline keyword parser when there is no API key). "open"/"close"/"let go" stay
on the gripper path.

## Cloud model + timeout (what still reaches Nemotron)

The cloud model is set by `nemotron.model` (switchable) and the client timeout by `nemotron.timeout_s` (default 6 s;
on a timeout the app replies "Sorry, I didn't catch that in time" instead of hanging). Benchmark 2026-10-23 on 20
tonight-style utterances that reach the cloud (objects / free phrasing / context / clarify), measuring per-call
latency and whether the returned Intent action matched the expected one (Nebius Token Factory endpoint; latencies
were elevated vs the historical 1.4-3.9 s, likely endpoint load, but the comparison is like-for-like):

| model | median | p90 | accuracy |
|-------|-------:|----:|---------:|
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B (previous default) | 4.48 s | 10.18 s | 18/20 |
| nvidia/Nemotron-3_5-Lightning | 4.78 s | 14.68 s | 19/20 |
| **Qwen/Qwen3-30B-A3B-Instruct-2507 (new default)** | **0.37 s** | **7.63 s** | **19/20** |

Qwen3-30B-A3B-Instruct is ~12x faster at the median with equal-or-better accuracy, so it is the new default (the
rule: switch only when accuracy is equal or better). Both other models' single miss and Qwen's single miss were the
same kind (answering "what's the weather" with clarify rather than unknown, and - for Nano only - reading "move it
up a little bit" as a whole-object move instead of a jog). All three returned valid structured JSON (0 schema errors).
