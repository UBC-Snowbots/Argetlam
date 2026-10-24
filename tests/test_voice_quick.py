"""Fast voice path (2026-10-23 real-mic test): short named-pose commands are parsed locally in ms (no cloud call),
and on dev_arm_real they move at once with no spoken "yes"; autonomous skills stay confirm-gated."""
import pytest

from rover_intent.config import load_config
from rover_intent.planner.nemotron import quick_pose

OBJ, PRE = ["cup", "bottle"], ["ready", "base"]


@pytest.mark.parametrize("text,want", [("Go to the ready pose.", "ready"), ("stow it", "base"), ("Stow.", "base"),
                                       ("go back", "return"), ("home", "return"), ("ready", "ready")])
def test_quick_pose_matches_short_pose_commands(text, want):
    q = quick_pose(text, OBJ, PRE)
    assert q is not None and q.preset == want


@pytest.mark.parametrize("text", ["put the cup back on the base", "go to ready then stow", "pick up the cup",
                                  "move left 5 cm", "could you please move the arm to the ready pose now",
                                  "what is the weather"])
def test_quick_pose_defers_everything_else_to_nemotron(text):
    assert quick_pose(text, OBJ, PRE) is None


def test_dev_arm_real_moves_poses_without_confirm_but_gates_skills():
    c = load_config(arm="dev_arm_real")["confirm"]
    assert c["presets"] is False and c["skills"] is True
    assert load_config(arm="dev_arm")["confirm"].get("presets", True) is True   # sim profile unchanged
