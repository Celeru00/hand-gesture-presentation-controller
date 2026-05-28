"""
gesture_engine.py
=================
Dev 2 — Gesture Recognition Engine

Responsibilities:
- Receive raw MediaPipe landmarks from Dev1 via receive_landmarks().
- Classify the hand shape and motion into a named gesture.
- Smooth results over multiple frames to avoid flickering.
- Send confirmed gesture events to Dev 3 via app.receive_gesture().

Does NOT:
- Open a webcam or video file.
- Run MediaPipe.
- Control the UI or send keyboard events.

How it connects:
    In main.py, after Dev 2 is ready:
        gesture = GestureEngine(app)
        vision.gesture_engine = gesture

    Dev 1 calls gesture.receive_landmarks(...) each frame.
    Dev 2 calls app.receive_gesture(event) when a gesture is confirmed.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Optional

from contracts import GestureEvent, SourceRequest


class GestureEngine:

    def __init__(self, app: Any) -> None:
        """
        Parameters
        ----------
        app:
            The PresentationControllerApp instance (Dev 3).
            GestureEngine calls app.receive_gesture(event) on confirmed gestures.
        """
        self._app = app

        # --- put your state variables here ---
        # e.g. rolling history buffer for smoothing, swipe tracking,
        # last gesture timestamp for cooldown, previous pinch distance, etc.

    # ------------------------------------------------------------------
    # Entry point — called by Dev 1 every frame
    # ------------------------------------------------------------------

    def receive_landmarks(
        self,
        landmarks: Any,
        frame_id: int,
        width: int,
        height: int,
    ) -> None:
        """
        Called by VisionEngine every frame with raw MediaPipe landmark data.

        Parameters
        ----------
        landmarks:
            mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            A list of 21 landmarks. Access each as landmarks.landmark[i]
            where i is the MediaPipe index (0 = wrist, 8 = index tip, etc.)
            Each landmark has .x, .y, .z normalized to [0.0, 1.0].

        frame_id:
            Incrementing frame counter from Dev 1.
            Use this to detect dropped frames if needed.

        width / height:
            Pixel dimensions of the source frame.
            Use these to convert normalized coords to pixel coords if needed:
                px = int(landmark.x * width)
                py = int(landmark.y * height)

        What to do here:
            1. Extract the landmarks you need (fingertips, wrist, etc.)
            2. Run finger state detection (extended vs folded)
            3. Check for swipe motion using a rolling wrist position buffer
            4. Run static pose classification on finger states
            5. Smooth the result over N frames
            6. If a stable gesture is confirmed, call self._emit(gesture_type, confidence)
            7. For pointer/draw gestures, also pass cursor_x and cursor_y
        """

        # --- your implementation here ---
        pass

    # ------------------------------------------------------------------
    # Internal helpers — implement these
    # ------------------------------------------------------------------

    def _get_finger_states(self, landmarks: Any) -> dict:
        """
        Determine which fingers are extended vs folded.

        For index, middle, ring, pinky:
            Compare fingertip y to PIP joint y.
            If tip.y < pip.y, the finger is extended (tip is higher on screen).

        For thumb:
            Lateral check — compare thumb tip x to thumb IP joint x.

        Returns
        -------
        dict with keys: "thumb", "index", "middle", "ring", "pinky"
        Values are True (extended) or False (folded).

        Landmark indices to use:
            Thumb:  tip=4,  ip=3
            Index:  tip=8,  pip=6
            Middle: tip=12, pip=10
            Ring:   tip=16, pip=14
            Pinky:  tip=20, pip=18
        """

        # --- your implementation here ---
        pass

    def _classify_static(self, finger_states: dict, landmarks: Any):
        """
        Map finger states to a gesture type string.

        Use the table below. Return (gesture_type, confidence).
        confidence is a float in [0.0, 1.0] — how certain you are.

        Gesture type strings must match what controller.py expects.
        Use normalize_gesture_name() aliases or the exact canonical names:
            "next_slide", "previous_slide",
            "start_presentation", "stop_exit",
            "blank_screen", "laser_pointer",
            "draw_annotate", "zoom_in", "zoom_out"

        Classification table (T=Thumb I=Index M=Middle R=Ring P=Pinky):

            T  I  M  R  P  → gesture
            1  1  1  1  1  → start_presentation   (open palm)
            0  0  0  0  0  → stop_exit            (fist)
            0  1  0  0  0  → laser_pointer         (index only)
            0  1  1  0  0  → draw_annotate         (peace sign)
            1  0  0  0  0  → blank_screen          (thumbs up)
            (pinch)        → zoom_in / zoom_out    (see _detect_pinch)
            (swipe)        → next_slide / prev_slide (see _detect_swipe)
            (else)         → unknown / no emit

        Returns
        -------
        tuple[str, float] — (gesture_type_string, confidence)
        """

        # --- your implementation here ---
        pass

    def _detect_swipe(self, landmarks: Any) -> Optional[str]:
        """
        Detect left/right swipe using wrist x-position over time.

        Approach:
            Keep a rolling buffer (deque) of the last N wrist x values.
            dx = buffer[-1] - buffer[0]
            If dx > threshold  → "next_slide"
            If dx < -threshold → "previous_slide"
            Apply a cooldown so one swipe doesn't fire multiple times.

        Wrist landmark index: 0
            wrist_x = landmarks.landmark[0].x   (normalized 0.0 to 1.0)

        Suggested values to tune:
            buffer length:  5 frames
            threshold:      0.12 (normalized units)
            cooldown:       0.8 seconds

        Returns
        -------
        gesture type string if swipe detected, None otherwise.
        """

        # --- your implementation here ---
        pass

    def _detect_pinch(self, landmarks: Any) -> Optional[str]:
        """
        Detect pinch open/close using thumb tip and index tip distance.

        Approach:
            Compute Euclidean distance between thumb tip (4) and index tip (8).
            Compare to the previous frame's distance stored in self.
            If distance increasing past threshold → "zoom_out"
            If distance decreasing past threshold → "zoom_in"

        Suggested threshold: 0.06 (normalized units)

        Returns
        -------
        gesture type string if pinch detected, None otherwise.
        """

        # --- your implementation here ---
        pass

    def _smooth(self, gesture_type: str) -> Optional[str]:
        """
        Reduce flickering by requiring a gesture to appear consistently
        across a rolling window of frames before emitting it.

        Approach:
            Keep a deque of the last N classified gesture strings.
            Only return a gesture if it appears in > 70% of the window.
            If the window is mixed, return None (not stable yet).

        Suggested window size: 5–8 frames.

        Returns
        -------
        The stable gesture type string, or None if not stable yet.
        """

        # --- your implementation here ---
        pass

    def _emit(
        self,
        gesture_type: str,
        confidence: float,
        cursor_x: Optional[float] = None,
        cursor_y: Optional[float] = None,
        frame_id: int = -1,
    ) -> None:
        """
        Send a confirmed gesture to Dev 3.

        Do NOT call this directly from receive_landmarks —
        always pass through _smooth() first so only stable gestures emit.

        cursor_x / cursor_y:
            Only needed for "laser_pointer" and "draw_annotate".
            Pass the normalized index finger tip x and y:
                cursor_x = landmarks.landmark[8].x
                cursor_y = landmarks.landmark[8].y

        Dev 3 handles everything after this — no further action needed.
        """

        event = GestureEvent(
            gesture_type=gesture_type,
            confidence=confidence,
            timestamp=time.time(),
            cursor_x=cursor_x,
            cursor_y=cursor_y,
            source_frame_id=frame_id,
        )
        self._app.receive_gesture(event)
