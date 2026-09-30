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
    elbow: Optional[tuple[float, float, float]] = None           # for pose mirroring
    shoulder_other: Optional[tuple[float, float, float]] = None  # the other shoulder (torso frame)
    hips: Optional[tuple[float, float, float]] = None            # hip midpoint (torso "up")
    side: Literal["right", "left"] = "right"
    wrist_other: Optional[tuple[float, float, float]] = None      # the other wrist (left-hand clutch)
    skel2d: Optional[list[tuple[float, float]]] = None            # normalized image points for the overlay skeleton
    seq: int = 0                                                  # for loss / latency measurement
    rtt_ms: Optional[float] = None                                # client-measured round trip, echoed to the HUD
    in_position: Optional[bool] = None                            # client's stand-here guide is green
    hand_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hand_open: Optional[float] = None  # 0 fist .. 1 open; None = hand not found (must NOT mean "open": it dropped objects)
    visible: bool = True


class Utterance(BaseModel):
    """Laptop -> VM: a finished speech-to-text transcript."""
    kind: Literal["utterance"] = "utterance"
    t: float
    text: str
    offline: bool = False  # True = keyword parser, no Nemotron (deterministic, for scripted checks)


class ClutchMsg(BaseModel):
    """Laptop -> VM: the F key (toggle the clutch)."""
    kind: Literal["clutch"] = "clutch"
    t: float = 0.0
    action: Literal["toggle", "warmup", "calibrate", "redo_axis"] = "toggle"


class Query(BaseModel):
    """-> VM: ask for the current state (mode, held object, object positions, gripper)."""
    kind: Literal["query"] = "query"
    t: float = 0.0


class EEGState(BaseModel):
    """EEG decoder output: probability that the user intends to grasp (motor imagery)."""
    kind: Literal["eeg"] = "eeg"
    t: float
    p_grasp: float
