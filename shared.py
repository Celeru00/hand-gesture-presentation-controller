"""
shared.py
============
Shared data contracts for the Hand Gesture Presentation Controller.

All three developers import from this file.
Standard library only — zero external dependencies.

Data flow:
    SourceRequest   Dev 3 → Dev 1   (change input source)
    FramePacket     Dev 1 → Dev 3   (raw annotated frame)
    GestureEvent    Dev 2 → Dev 3   (confirmed gesture)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True)
class SourceRequest:
    """
    Sent from Dev 3 (UI) to Dev 1 (Vision Engine).

    source_type:
        "webcam"     -> value is the camera index (int), usually 0
        "video_file" -> value is the file path (str)
        "stop"       -> value is None; stop all capture
    """

    source_type: str
    value: Optional[Union[int, str]] = None


@dataclass
class FramePacket:
    """
    Sent from Dev 1 (Vision Engine) to Dev 3 (UI).

    image:
        A numpy array. Dev 1 sends BGR format from OpenCV,
        with hand landmarks already drawn on the frame.

    color_format:
        "BGR"  — default OpenCV format
        "RGB"  — converted format
    """

    frame_id: int
    timestamp: float
    image: Any
    color_format: str = "BGR"
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class GestureEvent:
    """
    Sent from Dev 2 (Gesture Engine) to Dev 3 (UI + Controller).
    Only emitted when a gesture is stable and above the confidence threshold.

    cursor_x / cursor_y:
        Optional. Normalized coordinates in [0.0, 1.0].
        Only populated for LASER_POINTER and DRAW_ANNOTATE gestures.
        Dev 3 uses these to move the OS cursor.

    gesture_type accepted values (Dev 2 must use one of these strings,
    or any alias defined in controller.normalize_gesture_name):
        "next_slide", "previous_slide",
        "start_presentation", "stop_exit",
        "blank_screen", "laser_pointer",
        "draw_annotate", "zoom_in", "zoom_out"
    """

    gesture_type: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    cursor_x: Optional[float] = None
    cursor_y: Optional[float] = None
    source_frame_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
