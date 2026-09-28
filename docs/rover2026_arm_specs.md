# rover2026 Arm — Spec Sheet (derived)

**Sources:** `rover2026.urdf` (SolidWorks export, "MAIN Arm Assembly RevB 2025-2026 V6"), the 7 STL meshes,
40k-sample forward-kinematics sweep (`arm_specs.py`), and the Isaac Lab sim/training runs (Sept 2026).
**Frame:** base_link origin, z up, metres unless stated. **Date:** 2026-09-27.

> ⚠️ Treat masses/inertias as CAD estimates (SolidWorks material assignments) — weigh the real arm to confirm.
> The URDF has `effort="0" velocity="0"` on every joint, so **no motor torque/speed data exists here** —
> those must come from the motor/gearbox datasheets.

## Kinematic chain (6 DOF)

| # | Joint | Type | Axis role | Limits (°) | Range (°) | Offset from previous joint (mm) |
|---|---|---|---|---|---|---|
| 1 | `shoulder_joint` | revolute | base yaw (vertical) | −12.6 … 315.1 | 327.7 | 195.0 (base → shoulder) |
| 2 | `link_1_joint` | revolute | shoulder pitch | −179.9 … 0 | 179.9 | 87.5 |
| 3 | `link1_link2` | revolute | elbow | 0 … 179.9 | 179.9 | **504.0** (upper arm) |
| 4 | `a4_rotation` | revolute | forearm roll | −90 … 90 | 180 | 243.7 |
| 5 | `a5_rotation` | revolute | wrist | −179.9 … 179.9 | 359.8 | 242.9 |
| 6 | `a6_rotation` | **continuous** | EE roll | unlimited | 360+ | 107.3 |

**⚠️ Sim bug found:** `a6_rotation` is `continuous` in the URDF, but the exporter wrote `lower=0 upper=0`
and the Isaac URDF importer turned that into a **locked** joint. Every Isaac training run so far had the
EE roll frozen. Fix before the next run (set it as continuous / ±π in the USD or importer config) — it
likely also contributed to the solver fighting a "locked" joint (the NaN envs).

## Mass

| Link | Mass (kg) | Mesh bounding box (mm) |
|---|---|---|
| base_link | 0.764 | 160 × 260 × 104 |
| a1_shoulder_base | 1.216 | 127 × 137 × 167 |
| a2_link_1 (upper arm) | **2.378** | 602 × 109 × 138 |
| a3_axis4_housing | 0.406 | 96 × 199 × 173 |
| a4_link2 (forearm) | 1.695 | 485 × 106 × 160 |
| a5_internal_upright | 0.167 | 160 × 112 × 107 |
| a6_EE_holder | 0.379 | 91 × 110 × 110 |
| **Total** | **7.00** | |
| **Moving (excl. base)** | **6.24** | |

## Workspace (EE-holder frame origin)

- **Max reach from shoulder:** 1.18 m
- **Envelope:** x −1.05 … 1.21, y −0.95 … 1.30, **z −0.46 … 1.20** (matches the Isaac reachability probe)
- **Home pose (all joints 0):** EE at (−0.03, 0.15, 0.50)
- **Keypad** (sim placement from `keyboard_stack_v2/layout.py`, keys at z = 0.828, x 0.05–0.11, y −0.08…−0.04):
  every key reachable — closest random-sample approach 0.7–2.3 cm (sampling-limited; a policy/IK gets closer).
- Note: there is **no tool-tip link** — reach is measured to the EE-holder origin. A real presser/tool adds its
  length to reach and to the torques below.

## Static holding torque (gravity only, worst case over the sweep)

| Joint | Arm only (N·m) | + 0.5 kg at EE (N·m) |
|---|---|---|
| shoulder_joint (yaw) | 0 | 0 |
| **link_1_joint (shoulder pitch)** | **28.9** | **34.2** |
| **link1_link2 (elbow)** | **8.3** | **11.2** |
| a4_rotation | 0.3 | 0.8 |
| a5_rotation | 0.3 | 0.8 |
| a6_rotation | 0 | 0 |

Static only — no acceleration, friction or gearbox losses. **For motor sizing use ≥1.5–2× these**
(e.g. shoulder pitch ~50–70 N·m continuous at the output, elbow ~17–25 N·m).

## Sim / training facts (Isaac Lab, Sept 2026)

- Sim actuators used: implicit PD, stiffness 800 / damping 40 (v1) — caused NaN in some envs; v2 env uses
  150 / 25 with effort limit 60 N·m and velocity limit 8 rad/s (`isaac_migration/keypad_reach_env_v2.py`).
  These are **sim tuning values, not measured hardware specs**.
- PPO reach policy (1024 envs, 300 iters, ~6 min on A100): **93% of envs within 5 cm** of the target key,
  2.8% within 2 cm. Wrist roll was locked (see bug above).
- Throughput: ~21k env-steps/s in Isaac vs roughly 240 steps/s or more in the old RoboSuite setup.

## Open items for planning

1. Motor/gearbox datasheets → real torque & speed limits per joint (fill the URDF `effort`/`velocity`).
2. Weigh the arm to validate the 6.24 kg moving mass.
3. Define the EE tool (key presser) geometry + mass → adds reach and torque.
4. Measure the real keypad pose relative to the arm base (sim uses `layout.py` assumptions).
5. Fix the `a6_rotation` locked-joint import bug before retraining.
