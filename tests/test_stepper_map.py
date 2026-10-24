"""control.stepper_map: the single source of truth for Rowan's RAW-firmware-degrees <-> URDF-radians mapping."""
import math

import numpy as np

from rover_intent.control.stepper_map import DEFAULT_TABLE, StepperTable, fw_to_q, in_travel, q_to_fw


def test_matches_rowans_formula_and_table():
    # q = radians(fw)*dir + zero, per Rowan's firmToMoveitOffsetPos; the table is the OLD (physically-correct) one.
    assert DEFAULT_TABLE.dir == (1, 1, 1, 1, -1, -1)
    assert abs(DEFAULT_TABLE.zero_rad[0] - (-0.9608)) < 1e-9
    assert abs(DEFAULT_TABLE.zero_rad[4] - (2.2060 - math.pi / 3)) < 1e-9
    fw = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    q = fw_to_q(fw)
    for i in range(6):
        assert abs(q[i] - (math.radians(fw[i]) * DEFAULT_TABLE.dir[i] + DEFAULT_TABLE.zero_rad[i])) < 1e-12


def test_round_trip_fw_q_fw():
    fw = [12.3, 45.6, 78.9, 101.1, 33.3, 150.0]
    assert np.allclose(q_to_fw(fw_to_q(fw)), fw, atol=1e-9)


def test_fw_to_q_ignores_the_ee_and_takes_len7():
    fw7 = [10, 20, 30, 40, 50, 60, 170]   # index 6 = EE, not converted
    assert len(fw_to_q(fw7)) == 6
    assert np.allclose(fw_to_q(fw7), fw_to_q(fw7[:6]))


def test_table_is_a_parameter_reconvert_with_a_different_table():
    # same RAW degrees, a changed axis-4 offset (Rowan recalculating) -> different radians, as a parameter.
    fw = [0, 0, 0, 30.0, 0, 0]
    t2 = StepperTable(zero_rad=tuple(list(DEFAULT_TABLE.zero_rad[:3]) + [-2.0] + list(DEFAULT_TABLE.zero_rad[4:])))
    assert abs(fw_to_q(fw, t2)[3] - (math.radians(30.0) * 1 + (-2.0))) < 1e-12
    assert fw_to_q(fw, t2)[3] != fw_to_q(fw)[3]


def test_to_dict_from_dict_round_trip():
    d = DEFAULT_TABLE.to_dict()
    t = StepperTable.from_dict(d)
    assert t.zero_rad == DEFAULT_TABLE.zero_rad and t.dir == DEFAULT_TABLE.dir and t.version == DEFAULT_TABLE.version


def test_in_travel_respects_firmware_bounds_and_margin():
    assert in_travel([90, 90, 70, 90, 90, 90]) == [True] * 6
    assert in_travel([-1, 90, 70, 90, 90, 90]) == [False, True, True, True, True, True]  # below 0
    assert in_travel([181, 90, 70, 90, 90, 90]) == [False, True, True, True, True, True]  # above 180
    # axis 3 max 140; 138 fails with a 5 deg margin, passes with 0
    assert in_travel([90, 90, 138, 90, 90, 90], margin=5.0)[2] is False
    assert in_travel([90, 90, 138, 90, 90, 90], margin=0.0)[2] is True
