"""Unit tests for ros/servo/servo_to_arm_command.py: the Servo-velocity -> firmware 'V' mapping and the stale
timeout. Pure logic only (no ROS); a fake publisher stands in for the ArmCommand publisher."""
import importlib.util
import math
from pathlib import Path

import pytest

MOD = Path(__file__).resolve().parents[1] / "ros" / "servo" / "servo_to_arm_command.py"
spec = importlib.util.spec_from_file_location("servo_to_arm_command", MOD)
s2a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2a)

CAPS = (80.0, 40.0, 60.0, 110.0, 90.0, 90.0)
DIR = (1, 1, 1, 1, -1, -1)


def test_units_and_direction():
    m = s2a.ServoToArmMapper(speed=1.0)
    out = m.map_velocities([math.radians(10), 0, 0, 0, math.radians(10), math.radians(10)])
    # deg/s = degrees(rad) * dir; joints 5 and 6 flip sign
    assert out == pytest.approx([10, 0, 0, 0, -10, -10], abs=1e-9)


def test_speed_scaling():
    m = s2a.ServoToArmMapper(speed=0.2)
    out = m.map_velocities([math.radians(100), 0, 0, 0, 0, 0])
    assert out[0] == pytest.approx(20.0)        # 100 deg/s * 0.2


def test_clamped_to_firmware_caps_with_sign():
    m = s2a.ServoToArmMapper(speed=1.0)
    out = m.map_velocities([math.radians(500), -math.radians(500), 0, 0, math.radians(500), 0])
    assert out[0] == CAPS[0]                     # +cap on joint 1
    assert out[1] == -CAPS[1]                    # -cap on joint 2
    assert out[4] == -CAPS[4]                    # dir -1: +input clamps to -cap
    assert all(abs(v) <= c + 1e-9 for v, c in zip(out, CAPS))


def test_cmd_type_is_V_and_six_joints():
    m = s2a.ServoToArmMapper()
    cmd = m.on_trajectory([0.1] * 6, now=0.0)
    assert cmd.cmd_type == ord("V") == 86 and len(cmd.velocities) == 6


def test_wrong_joint_count_raises():
    m = s2a.ServoToArmMapper()
    with pytest.raises(ValueError):
        m.map_velocities([0.1, 0.2, 0.3])


def test_zero_command():
    z = s2a.ServoToArmMapper().zero_command()
    assert z.cmd_type == 86 and z.velocities == [0.0] * 6


def test_stale_timeout_sends_one_zero_then_resets():
    class FakePub:
        def __init__(self):
            self.sent = []

        def publish(self, cmd):
            self.sent.append(cmd)

    fp = FakePub()
    m = s2a.ServoToArmMapper(speed=0.2, timeout=0.2)
    fp.publish(m.on_trajectory([math.radians(50)] + [0] * 5, now=0.0))
    assert m.poll(0.1) is None                   # fresh: no halt
    stale = m.poll(0.25)                          # > timeout since last command
    assert stale is not None and stale.velocities == [0.0] * 6 and stale.cmd_type == 86
    fp.publish(stale)
    assert m.poll(0.30) is None                   # already stopped: no repeated zeros
    # a new command re-arms the timeout
    fp.publish(m.on_trajectory([math.radians(50)] + [0] * 5, now=0.4))
    assert m.poll(0.45) is None
    assert m.poll(0.65) is not None
    assert [c.cmd_type for c in fp.sent] == [86, 86, 86]     # all 'V'
    assert fp.sent[0].velocities[0] == pytest.approx(10.0)   # 50 deg/s * 0.2
