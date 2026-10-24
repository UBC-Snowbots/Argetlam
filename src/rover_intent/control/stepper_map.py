"""Single source of truth for Rowan's Stepper mapping between RAW firmware degrees and our URDF joint radians.

Rowan's stepper_arm_driver (RoverFlake2) publishes /arm/feedback as RAW firmware degrees (step counts converted
to degrees by the firmware; open-loop, so this is commanded steps, not a measured encoder angle). It also
publishes /joint_states, but the driver currently loads ArmConstants::axis_zero_rads = all 0.0 and axis_dirs =
all 1 (RoverFlake2 ArmControlParamsStepper.h), so /joint_states carries ZERO offsets and is WRONG for our URDF.
We therefore ignore /joint_states and convert /arm/feedback ourselves with the real table below.

Per axis (the driver's own firmToMoveitOffsetPos: `rad*dir + zero_rad`):

    q_rad  = radians(fw_deg) * dir + zero_rad
    fw_deg = degrees((q_rad - zero_rad) * dir)        # dir is +-1, so dividing == multiplying

The table values are Rowan's ArmConstants::OLD_axis_zero_rads / OLD_axis_dirs (the physically-correct offsets;
the active in-driver table is all zeros, see above). KEEP RAW firmware degrees in every recording and convert
only at read time: axis 4's gear ratio is known wrong (Rowan recalculating) and the offset table may change, so
a recording stays re-convertible by passing a new StepperTable.

Pure stdlib (math + dataclasses) so this imports on the system python the ROS nodes use (follower, real_to_sim)
as well as inside the venv app. No numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Bump this when the numbers below change. meta.json records it so a CSV can say which table produced its radians.
TABLE_VERSION = "rowan-stepper-2026-10-23"

# Note vs firmware: RoverFlake2 defines OLD axis-5 zero as 2.2060 - PI/3 with its own PI = 3.14 (= 1.15933 rad).
# We use math.pi (= 1.15880 rad), matching the sim_to_real follower. The 0.0005 rad (~0.03 deg) gap is noise.
_AXIS5_ZERO = 2.2060 - math.pi / 3


@dataclass(frozen=True)
class StepperTable:
    """Rowan's Stepper mapping for the 6 arm axes (index 0..5; the EE at firmware index 6 is not converted)."""
    zero_rad: tuple = (-0.9608, -1.9390, -1.3460, -2.4108, _AXIS5_ZERO, 0.0)
    dir: tuple = (1, 1, 1, 1, -1, -1)
    fw_max_deg: tuple = (180.0, 180.0, 140.0, 180.0, 180.0, 9999.0)   # firmware travel [0, max] per axis
    fw_max_vel: tuple = (80.0, 40.0, 60.0, 110.0, 90.0, 90.0)         # firmware max speed, deg/s
    version: str = TABLE_VERSION

    def to_dict(self) -> dict:
        return {"zero_rad": list(self.zero_rad), "dir": list(self.dir),
                "fw_max_deg": list(self.fw_max_deg), "fw_max_vel": list(self.fw_max_vel), "version": self.version}

    @classmethod
    def from_dict(cls, d: dict) -> "StepperTable":
        """Rebuild a table from meta.json so a recording can be re-converted with the table it was taken under."""
        return cls(zero_rad=tuple(d["zero_rad"]), dir=tuple(d["dir"]),
                   fw_max_deg=tuple(d.get("fw_max_deg", cls.fw_max_deg)),
                   fw_max_vel=tuple(d.get("fw_max_vel", cls.fw_max_vel)),
                   version=d.get("version", "unknown"))


DEFAULT_TABLE = StepperTable()


def fw_to_q(fw_deg, table: StepperTable = DEFAULT_TABLE) -> list:
    """RAW firmware degrees -> URDF joint radians, for the 6 arm axes. fw_deg may be length 6 or 7 (EE ignored)."""
    return [math.radians(fw_deg[i]) * table.dir[i] + table.zero_rad[i] for i in range(6)]


def q_to_fw(q_rad, table: StepperTable = DEFAULT_TABLE) -> list:
    """URDF joint radians -> RAW firmware degrees, for the 6 arm axes."""
    return [math.degrees((q_rad[i] - table.zero_rad[i]) * table.dir[i]) for i in range(6)]


def in_travel(fw_deg, table: StepperTable = DEFAULT_TABLE, margin: float = 0.0) -> list:
    """Per-axis: is fw_deg[i] within [margin, fw_max_deg[i] - margin]? (the firmware travel, 0 at the limit switch)."""
    return [margin <= float(fw_deg[i]) <= table.fw_max_deg[i] - margin for i in range(6)]


# Measured on the bench 2026-10-23 (scripts/bench/axis_sweep.py, 5/10/16 deg/s steps, fitted steady slope of the RAW
# /arm/feedback counts): every axis runs at roughly HALF the commanded 'V' speed (firmware two-phase pulse generation,
# see docs/REAL_DATA.md). Time from a command to the first count change: 0.17-0.31 s; the ramp is near-instant after.
# Use: actual_speed ~= MEASURED_VEL_RATIO[axis] * commanded. Not applied to any command (the follower is closed-loop);
# it sets the sim speed limits in dev_arm_real so the sim can't outrun the real arm.
MEASURED_VEL_RATIO = (0.486, 0.447, 0.472, 0.422, 0.490, 0.442)
MEASURED_CMD_LAG_S = (0.27, 0.19, 0.22, 0.17, 0.31, 0.18)
