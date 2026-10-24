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
    preset = "preset"      # move to a named joint-space pose from the arm profile (presets.py); needs confirm
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
    preset: Optional[str] = Field(None, description="named pose for action 'preset' (e.g. 'base', 'ready', 'return')")
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


class ArmPreset(BaseModel):
    """Body-client button: move the arm to a named pose ("return", a profile preset) or, with confirm, home the
    real arm ("home_hw"). Preset moves run through the SafetyLayer in HOLD and only while the clutch is idle."""
    kind: Literal["preset"] = "preset"
    t: float
    name: str
    confirm: bool = False


class GripToggle(BaseModel):
    """One-shot gripper toggle (close if open, open if closed), e.g. a Muse 2 double jaw clench (eeg/clench.py).
    An event, not a level, so it can't fight voice / reset / assist changes to the gripper."""
    kind: Literal["grip_toggle"] = "grip_toggle"
    t: float
    source: str = "clench"


class CameraFrame(BaseModel):
    """Laptop (table camera) -> VM: one JPEG frame for table-plane perception.

    The JPEG is base64-encoded so the frame rides the same JSON transport as every other message
    (UDP, or newline-JSON over the SSH tunnel). Detection + projection run on the VM GPU; the laptop
    only captures and sends. `seq`/`t` are for latency and drop accounting like BodyPose."""
    kind: Literal["camera_frame"] = "camera_frame"
    t: float
    jpeg_b64: str
    width: int = 0
    height: int = 0
    seq: int = 0
    source: str = "table_cam"
