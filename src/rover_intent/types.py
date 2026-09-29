"""Shared message types. These also define the JSON exchanged between the laptop and the VM."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Action(str, Enum):
    pick = "pick"          # grasp an object
    place = "place"        # put the held object relative to a reference
    move = "move"          # pick + place in one command
    grasp = "grasp"        # close the gripper where it is
    release = "release"    # open the gripper where it is
    home = "home"
    stop = "stop"
    follow = "follow"      # hand control back to body mirroring
    reset = "reset"        # arm home, gripper open, objects back to their start poses (sim)
    clarify = "clarify"    # ambiguous / mismatched description: ask the user (question in `say`), don't move
    unknown = "unknown"


class Intent(BaseModel):
    """Structured output from Nemotron (or the offline fallback parser)."""
    action: Action
    object: Optional[str] = Field(None, description="object to act on, e.g. 'cup'")
    relation: Optional[Literal["beside", "left_of", "right_of", "in_front_of", "behind", "on", "into"]] = None
    reference: Optional[str] = Field(None, description="reference object for the relation, e.g. 'bottle'")
    say: str = Field("", description="short spoken confirmation for the user")


class BodyPose(BaseModel):
    """Laptop -> VM: human wrist pose from MediaPipe, in the camera/shoulder frame (metres)."""
    kind: Literal["body"] = "body"
    t: float
    wrist: tuple[float, float, float]
    shoulder: tuple[float, float, float]
    hand_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hand_open: float = 1.0  # 0 closed .. 1 open (for a visual check vs EEG)
    visible: bool = True


class Utterance(BaseModel):
    """Laptop -> VM: a finished speech-to-text transcript."""
    kind: Literal["utterance"] = "utterance"
    t: float
    text: str


class EEGState(BaseModel):
    """EEG decoder output: probability that the user intends to grasp (motor imagery)."""
    kind: Literal["eeg"] = "eeg"
    t: float
    p_grasp: float
