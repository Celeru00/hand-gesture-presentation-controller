"""
Responsibilities:
- Receive 21 hand landmarks from Person 1.
- Classify the hand shape or movement into a known gesture.
- Smooth noisy results so gestures do not flicker.
- Emit confirmed GestureEvent objects to Person 3.

This file deliberately does NOT:
- Open webcam or video files.
- Run MediaPipe.
- Build a UI.
- Send keyboard shortcuts.
- Control PowerPoint.
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .shared import GestureEvent, LandmarkPacket, LandmarkPoint


# ---------------------------------------------------------------------
# Canonical gesture names expected by Person 3
# ---------------------------------------------------------------------

NEXT_SLIDE = "next_slide"
PREVIOUS_SLIDE = "previous_slide"
START_PRESENTATION = "start_presentation"
STOP_EXIT = "stop_exit"
BLANK_SCREEN = "blank_screen"
LASER_POINTER = "laser_pointer"
ZOOM_IN = "zoom_in"
ZOOM_OUT = "zoom_out"

POINTER_GESTURES = {LASER_POINTER}

DISCRETE_STATIC_GESTURES = {
    START_PRESENTATION,
    STOP_EXIT,
    BLANK_SCREEN,
}

DYNAMIC_GESTURES = {
    NEXT_SLIDE,
    PREVIOUS_SLIDE,
    ZOOM_IN,
    ZOOM_OUT,
}


class HandLandmark(IntEnum):
    """
    MediaPipe hand landmark index order.
    """

    WRIST = 0

    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class GestureRecognitionConfig:
    """
    Tune these values depending on camera placement, distance, lighting,
    and whether Person 1 mirrors the video feed.
    """

    # Static gesture confirmation
    min_static_confidence: float = 0.62
    stable_frames: int = 5
    stable_ratio: float = 0.70
    unstable_reset_frames: int = 4

    # Swipe detection
    swipe_history_seconds: float = 0.65
    swipe_min_dx: float = 0.18
    swipe_max_dy: float = 0.14
    swipe_min_velocity: float = 0.55
    swipe_cooldown_seconds: float = 0.90
    # The webcam feed is mirrored (you see yourself like a mirror), so when
    # the user physically swipes to their right, the hand in the image moves
    # to the LEFT. Mirroring dx makes the gesture map to the natural direction:
    # physical-right swipe → next slide, physical-left swipe → previous slide.
    mirror_swipe_x: bool = True

    # Pinch zoom detection — step-based.
    # We track the thumb-index distance and fire a zoom event each time it
    # changes by pinch_step_distance from the last baseline. This lets the
    # user keep zooming in by repeatedly spreading their fingers, without
    # having to alternate closed/open like the old transition model.
    pinch_step_distance: float = 0.30
    pinch_cooldown_seconds: float = 0.20
    # Minimum "pinch-shape" score required to start tracking. Lower = more
    # forgiving (engages with looser hand poses).
    pinch_shape_min_score: float = 0.45
    # After firing a zoom, opposite-direction firing is suppressed for this
    # long. Lets the user "reset" their hand position by closing fingers
    # without accidentally firing zoom_out between zoom_ins. Suppression
    # auto-refreshes while reverse motion is active.
    pinch_reverse_suppression_seconds: float = 0.6

    # Pointer smoothing
    pointer_filter_min_cutoff: float = 1.0
    pointer_filter_beta: float = 0.007
    pointer_filter_derivative_cutoff: float = 1.0

    # Event sending
    emit_dict_to_callback: bool = True

    # Cooldowns for static one-shot gestures.
    # Pointer gestures are intentionally not cooled down because Person 3
    # needs continuous cursor updates.
    discrete_static_cooldowns: Dict[str, float] = field(
        default_factory=lambda: {
            START_PRESENTATION: 2.00,
            STOP_EXIT: 1.50,
            BLANK_SCREEN: 1.00,
        }
    )

    # Hold-to-confirm times. The user must keep the pose stable for this
    # many seconds before the action fires. Prevents accidental triggers
    # on destructive actions like exiting the slideshow.
    hold_seconds: Dict[str, float] = field(
        default_factory=lambda: {
            STOP_EXIT: 3.0,
        }
    )


@dataclass
class GestureCandidate:
    gesture_type: str
    confidence: float
    cursor_x: Optional[float] = None
    cursor_y: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandFeatures:
    landmarks: List[LandmarkPoint]

    palm_x: float
    palm_y: float
    hand_scale: float

    finger_extended: Dict[str, float]
    finger_folded: Dict[str, float]

    thumb_open_score: float
    thumb_up_score: float

    index_middle_separation: float
    index_pinky_separation: float
    pinch_distance: float

    index_cursor_x: float
    index_cursor_y: float


# ---------------------------------------------------------------------
# One Euro Filter
# ---------------------------------------------------------------------

class LowPassFilter:
    def __init__(self) -> None:
        self.initialized = False
        self.previous_raw_value = 0.0
        self.previous_filtered_value = 0.0

    def filter(self, value: float, alpha: float) -> float:
        if not self.initialized:
            self.initialized = True
            self.previous_raw_value = value
            self.previous_filtered_value = value
            return value

        filtered = alpha * value + (1.0 - alpha) * self.previous_filtered_value
        self.previous_raw_value = value
        self.previous_filtered_value = filtered
        return filtered


class OneEuroFilter:
    """
    Small dependency-free implementation of the 1€ filter.

    It smooths slow jitter strongly, while allowing faster movement to pass
    with less lag.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        derivative_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_timestamp: Optional[float] = None

    def reset(self) -> None:
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_timestamp = None

    def filter(self, value: float, timestamp: float) -> float:
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return self.x_filter.filter(value, 1.0)

        dt = max(timestamp - self.last_timestamp, 1e-6)
        self.last_timestamp = timestamp

        previous_value = self.x_filter.previous_raw_value
        derivative = (value - previous_value) / dt

        derivative_alpha = self._alpha(dt, self.derivative_cutoff)
        smoothed_derivative = self.dx_filter.filter(derivative, derivative_alpha)

        cutoff = self.min_cutoff + self.beta * abs(smoothed_derivative)
        alpha = self._alpha(dt, cutoff)

        return self.x_filter.filter(value, alpha)

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


# ---------------------------------------------------------------------
# Gesture engine
# ---------------------------------------------------------------------

class GestureRecognitionEngine:
    """
    Main Person 2 class.

    Typical connection:

        gesture_engine = GestureRecognitionEngine(
            on_gesture=ui.receive_gesture
        )

        # Person 1 calls this every frame:
        gesture_engine.process_landmarks(landmark_packet)
    """

    def __init__(
        self,
        on_gesture: Optional[Callable[[Union[GestureEvent, Dict[str, Any]]], None]] = None,
        config: Optional[GestureRecognitionConfig] = None,
    ) -> None:
        self.on_gesture = on_gesture
        self.config = config or GestureRecognitionConfig()

        self._static_window: Deque[GestureCandidate] = deque(
            maxlen=self.config.stable_frames
        )
        self._unstable_frames = 0
        self._active_static_name: Optional[str] = None
        self._active_static_emitted = False
        self._active_static_started_at: Optional[float] = None

        self._motion_history: Deque[Tuple[float, float, float]] = deque()
        self._last_emitted_at: Dict[str, float] = {}

        # Step-based pinch tracking
        self._pinch_baseline_distance: Optional[float] = None
        self._last_pinch_gesture: Optional[str] = None
        self._pinch_reverse_until: float = 0.0

        self._pointer_x_filter = OneEuroFilter(
            min_cutoff=self.config.pointer_filter_min_cutoff,
            beta=self.config.pointer_filter_beta,
            derivative_cutoff=self.config.pointer_filter_derivative_cutoff,
        )
        self._pointer_y_filter = OneEuroFilter(
            min_cutoff=self.config.pointer_filter_min_cutoff,
            beta=self.config.pointer_filter_beta,
            derivative_cutoff=self.config.pointer_filter_derivative_cutoff,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def process_landmarks(
        self,
        packet: Union[LandmarkPacket, Mapping[str, Any]],
    ) -> Optional[GestureEvent]:
        """
        Process one frame of hand landmarks.

        Returns:
            GestureEvent if a stable gesture is confirmed.
            None otherwise.

        If self.on_gesture is set, the event is also sent to that callback.
        """

        landmark_packet = self._coerce_landmark_packet(packet)

        if len(landmark_packet.landmarks) != 21:
            self.process_no_hand()
            return None

        features = self._extract_features(landmark_packet.landmarks)

        self._update_motion_history(
            timestamp=landmark_packet.timestamp,
            palm_x=features.palm_x,
            palm_y=features.palm_y,
        )

        static_candidate = self._classify_static_gesture(features)

        # Pointer and draw gestures intentionally suppress swipe detection.
        # Otherwise moving the pointer could accidentally become a swipe.
        if static_candidate is None or static_candidate.gesture_type not in POINTER_GESTURES:
            swipe_event = self._detect_swipe(landmark_packet, features)
            if swipe_event is not None:
                self._reset_static_stability()
                self._emit(swipe_event)
                return swipe_event

        pinch_event = self._detect_pinch_transition(landmark_packet, features)
        if pinch_event is not None:
            self._reset_static_stability()
            self._emit(pinch_event)
            return pinch_event

        static_event = self._stabilize_static_candidate(
            packet=landmark_packet,
            candidate=static_candidate,
        )

        if static_event is not None:
            self._emit(static_event)
            return static_event

        return None

    def process_no_hand(self) -> None:
        """
        Call this when Person 1 has no hand landmarks for the current frame.
        """

        self._reset_static_stability()
        self._motion_history.clear()
        self._pinch_baseline_distance = None
        self._last_pinch_gesture = None
        self._pinch_reverse_until = 0.0
        self._pointer_x_filter.reset()
        self._pointer_y_filter.reset()

    # Friendly aliases
    update = process_landmarks
    process = process_landmarks

    # -----------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------

    def _extract_features(self, lms: List[LandmarkPoint]) -> HandFeatures:
        wrist = lms[HandLandmark.WRIST]

        palm_points = [
            lms[HandLandmark.WRIST],
            lms[HandLandmark.INDEX_MCP],
            lms[HandLandmark.MIDDLE_MCP],
            lms[HandLandmark.RING_MCP],
            lms[HandLandmark.PINKY_MCP],
        ]

        palm_x = sum(p.x for p in palm_points) / len(palm_points)
        palm_y = sum(p.y for p in palm_points) / len(palm_points)

        # Palm scale keeps thresholds independent of distance from camera.
        hand_scale = max(
            self._distance(wrist, lms[HandLandmark.MIDDLE_MCP]),
            self._distance(lms[HandLandmark.INDEX_MCP], lms[HandLandmark.PINKY_MCP]),
            1e-6,
        )

        index_ext = self._long_finger_extended_score(
            lms, HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP,
            HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP, hand_scale
        )
        middle_ext = self._long_finger_extended_score(
            lms, HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP,
            HandLandmark.MIDDLE_DIP, HandLandmark.MIDDLE_TIP, hand_scale
        )
        ring_ext = self._long_finger_extended_score(
            lms, HandLandmark.RING_MCP, HandLandmark.RING_PIP,
            HandLandmark.RING_DIP, HandLandmark.RING_TIP, hand_scale
        )
        pinky_ext = self._long_finger_extended_score(
            lms, HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP,
            HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP, hand_scale
        )

        thumb_open = self._thumb_open_score(lms, hand_scale)
        thumb_up = self._thumb_up_score(lms, hand_scale, thumb_open)

        finger_extended = {
            "thumb": thumb_open,
            "index": index_ext,
            "middle": middle_ext,
            "ring": ring_ext,
            "pinky": pinky_ext,
        }

        finger_folded = {
            "thumb": 1.0 - thumb_open,
            "index": self._long_finger_folded_score(
                lms, HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP,
                HandLandmark.INDEX_TIP, index_ext, hand_scale
            ),
            "middle": self._long_finger_folded_score(
                lms, HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP,
                HandLandmark.MIDDLE_TIP, middle_ext, hand_scale
            ),
            "ring": self._long_finger_folded_score(
                lms, HandLandmark.RING_MCP, HandLandmark.RING_PIP,
                HandLandmark.RING_TIP, ring_ext, hand_scale
            ),
            "pinky": self._long_finger_folded_score(
                lms, HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP,
                HandLandmark.PINKY_TIP, pinky_ext, hand_scale
            ),
        }

        index_middle_separation = (
            self._distance(lms[HandLandmark.INDEX_TIP], lms[HandLandmark.MIDDLE_TIP])
            / hand_scale
        )

        index_pinky_separation = (
            self._distance(lms[HandLandmark.INDEX_TIP], lms[HandLandmark.PINKY_TIP])
            / hand_scale
        )

        pinch_distance = (
            self._distance(lms[HandLandmark.THUMB_TIP], lms[HandLandmark.INDEX_TIP])
            / hand_scale
        )

        index_tip = lms[HandLandmark.INDEX_TIP]

        return HandFeatures(
            landmarks=lms,
            palm_x=palm_x,
            palm_y=palm_y,
            hand_scale=hand_scale,
            finger_extended=finger_extended,
            finger_folded=finger_folded,
            thumb_open_score=thumb_open,
            thumb_up_score=thumb_up,
            index_middle_separation=index_middle_separation,
            index_pinky_separation=index_pinky_separation,
            pinch_distance=pinch_distance,
            index_cursor_x=index_tip.x,
            index_cursor_y=index_tip.y,
        )

    def _long_finger_extended_score(
        self,
        lms: List[LandmarkPoint],
        mcp_i: int,
        pip_i: int,
        dip_i: int,
        tip_i: int,
        scale: float,
    ) -> float:
        wrist = lms[HandLandmark.WRIST]
        mcp = lms[mcp_i]
        pip = lms[pip_i]
        dip = lms[dip_i]
        tip = lms[tip_i]

        tip_wrist = self._distance(tip, wrist)
        pip_wrist = self._distance(pip, wrist)

        # Extended fingers usually have the tip farther from wrist than PIP.
        distance_score = self._score_range((tip_wrist - pip_wrist) / scale, 0.05, 0.45)

        # For this project, index/open palm are meant to face upward.
        # In image coordinates, smaller y means higher on screen.
        vertical_score = self._score_range((pip.y - tip.y) / scale, 0.08, 0.55)

        # Straightness from MCP-PIP-DIP-TIP chain.
        angle_1 = self._angle_degrees(mcp, pip, dip)
        angle_2 = self._angle_degrees(pip, dip, tip)
        straightness_score = (
            self._score_range(angle_1, 135.0, 175.0)
            + self._score_range(angle_2, 135.0, 175.0)
        ) / 2.0

        return self._clamp01(
            0.45 * distance_score
            + 0.30 * vertical_score
            + 0.25 * straightness_score
        )

    def _long_finger_folded_score(
        self,
        lms: List[LandmarkPoint],
        mcp_i: int,
        pip_i: int,
        tip_i: int,
        extended_score: float,
        scale: float,
    ) -> float:
        mcp = lms[mcp_i]
        pip = lms[pip_i]
        tip = lms[tip_i]

        tip_close_to_mcp = 1.0 - self._score_range(
            self._distance(tip, mcp) / scale,
            0.55,
            1.35,
        )

        tip_not_above_pip = 1.0 - self._score_range(
            (pip.y - tip.y) / scale,
            0.05,
            0.40,
        )

        return self._clamp01(
            0.60 * (1.0 - extended_score)
            + 0.25 * tip_close_to_mcp
            + 0.15 * tip_not_above_pip
        )

    def _thumb_open_score(
        self,
        lms: List[LandmarkPoint],
        scale: float,
    ) -> float:
        thumb_mcp = lms[HandLandmark.THUMB_MCP]
        thumb_ip = lms[HandLandmark.THUMB_IP]
        thumb_tip = lms[HandLandmark.THUMB_TIP]
        index_mcp = lms[HandLandmark.INDEX_MCP]

        thumb_length_score = self._score_range(
            self._distance(thumb_tip, thumb_mcp) / scale,
            0.45,
            1.00,
        )

        thumb_away_from_palm_score = self._score_range(
            self._distance(thumb_tip, index_mcp) / scale,
            0.45,
            1.05,
        )

        thumb_straight_score = self._score_range(
            self._angle_degrees(thumb_mcp, thumb_ip, thumb_tip),
            125.0,
            170.0,
        )

        return self._clamp01(
            0.40 * thumb_length_score
            + 0.40 * thumb_away_from_palm_score
            + 0.20 * thumb_straight_score
        )

    def _thumb_up_score(
        self,
        lms: List[LandmarkPoint],
        scale: float,
        thumb_open_score: float,
    ) -> float:
        wrist = lms[HandLandmark.WRIST]
        thumb_mcp = lms[HandLandmark.THUMB_MCP]
        thumb_tip = lms[HandLandmark.THUMB_TIP]
        index_mcp = lms[HandLandmark.INDEX_MCP]

        tip_above_mcp = self._score_range(
            (thumb_mcp.y - thumb_tip.y) / scale,
            0.15,
            0.80,
        )

        tip_above_index_base = self._score_range(
            (index_mcp.y - thumb_tip.y) / scale,
            0.10,
            0.70,
        )

        tip_above_wrist = self._score_range(
            (wrist.y - thumb_tip.y) / scale,
            0.10,
            0.80,
        )

        return self._clamp01(
            0.35 * thumb_open_score
            + 0.30 * tip_above_mcp
            + 0.20 * tip_above_index_base
            + 0.15 * tip_above_wrist
        )

    # -----------------------------------------------------------------
    # Static gesture classification
    # -----------------------------------------------------------------

    def _classify_static_gesture(
        self,
        f: HandFeatures,
    ) -> Optional[GestureCandidate]:
        ext = f.finger_extended
        fold = f.finger_folded

        open_spread_score = self._score_range(f.index_pinky_separation, 0.80, 1.80)

        open_palm_score = self._mean(
            ext["thumb"],
            ext["index"],
            ext["middle"],
            ext["ring"],
            ext["pinky"],
            open_spread_score,
        )

        fist_score = self._mean(
            fold["thumb"],
            fold["index"],
            fold["middle"],
            fold["ring"],
            fold["pinky"],
        )

        thumbs_up_score = self._mean(
            f.thumb_up_score,
            fold["index"],
            fold["middle"],
            fold["ring"],
            fold["pinky"],
        )

        # When the thumb tip is close to the index tip, the hand is in a
        # pinch shape — pinch detection should own that gesture, not the
        # static classifier. Penalize laser to prevent misclassification.
        not_pinching_score = self._score_range(f.pinch_distance, 0.70, 1.20)

        index_only_score = self._mean(
            ext["index"],
            fold["thumb"],
            fold["middle"],
            fold["ring"],
            fold["pinky"],
            not_pinching_score,
        )

        candidates = [
            GestureCandidate(
                gesture_type=START_PRESENTATION,
                confidence=open_palm_score,
                metadata={"reason": "open palm"},
            ),
            GestureCandidate(
                gesture_type=STOP_EXIT,
                confidence=fist_score,
                metadata={"reason": "closed fist"},
            ),
            GestureCandidate(
                gesture_type=BLANK_SCREEN,
                confidence=thumbs_up_score,
                metadata={"reason": "thumbs up"},
            ),
            GestureCandidate(
                gesture_type=LASER_POINTER,
                confidence=index_only_score,
                cursor_x=f.index_cursor_x,
                cursor_y=f.index_cursor_y,
                metadata={"reason": "index finger only"},
            ),
        ]

        best = max(candidates, key=lambda c: c.confidence)

        if best.confidence < self.config.min_static_confidence:
            return None

        best.metadata.update(
            {
                "finger_extended": dict(ext),
                "finger_folded": dict(fold),
                "pinch_distance": f.pinch_distance,
                "hand_scale": f.hand_scale,
            }
        )

        return best

    # -----------------------------------------------------------------
    # Static gesture stabilization
    # -----------------------------------------------------------------

    def _stabilize_static_candidate(
        self,
        packet: LandmarkPacket,
        candidate: Optional[GestureCandidate],
    ) -> Optional[GestureEvent]:
        if candidate is None:
            self._unstable_frames += 1

            if self._unstable_frames >= self.config.unstable_reset_frames:
                self._reset_static_stability()

            return None

        self._unstable_frames = 0
        self._static_window.append(candidate)

        if len(self._static_window) < self.config.stable_frames:
            return None

        names = [c.gesture_type for c in self._static_window]
        counts = Counter(names)
        stable_name, count = counts.most_common(1)[0]

        ratio = count / len(self._static_window)

        if ratio < self.config.stable_ratio:
            return None

        stable_candidates = [
            c for c in self._static_window
            if c.gesture_type == stable_name
        ]

        average_confidence = self._mean(*(c.confidence for c in stable_candidates))

        if average_confidence < self.config.min_static_confidence:
            return None

        latest_same = stable_candidates[-1]

        if stable_name != self._active_static_name:
            self._active_static_name = stable_name
            self._active_static_emitted = False
            self._active_static_started_at = packet.timestamp

        # Pointer gestures need continuous events, because Person 3 moves
        # or drags the cursor based on these coordinates.
        if stable_name in POINTER_GESTURES:
            cursor_x, cursor_y = self._smooth_cursor(
                latest_same.cursor_x,
                latest_same.cursor_y,
                packet.timestamp,
            )

            return GestureEvent(
                gesture_type=stable_name,
                confidence=average_confidence,
                timestamp=packet.timestamp,
                cursor_x=cursor_x,
                cursor_y=cursor_y,
                source_frame_id=packet.frame_id,
                metadata=dict(latest_same.metadata),
            )

        # One-shot static gestures should emit once per hold.
        if self._active_static_emitted:
            return None

        # Hold-to-confirm: require the user to keep the pose stable for N
        # seconds before firing. While holding, emit progress events so
        # the UI can show feedback without triggering the real action.
        required_hold = self.config.hold_seconds.get(stable_name, 0.0)
        if required_hold > 0.0:
            started_at = self._active_static_started_at or packet.timestamp
            held_for = packet.timestamp - started_at
            if held_for < required_hold:
                progress = max(0.0, min(1.0, held_for / required_hold))
                return GestureEvent(
                    gesture_type=stable_name,
                    confidence=average_confidence,
                    timestamp=packet.timestamp,
                    source_frame_id=packet.frame_id,
                    metadata={
                        **dict(latest_same.metadata),
                        "hold_in_progress": True,
                        "hold_progress": progress,
                        "hold_required_seconds": required_hold,
                        "hold_elapsed_seconds": held_for,
                    },
                )

        if not self._cooldown_ok(stable_name, packet.timestamp):
            return None

        self._active_static_emitted = True
        self._last_emitted_at[stable_name] = packet.timestamp

        return GestureEvent(
            gesture_type=stable_name,
            confidence=average_confidence,
            timestamp=packet.timestamp,
            source_frame_id=packet.frame_id,
            metadata=dict(latest_same.metadata),
        )

    def _reset_static_stability(self) -> None:
        self._static_window.clear()
        self._unstable_frames = 0
        self._active_static_name = None
        self._active_static_emitted = False
        self._active_static_started_at = None
        self._pointer_x_filter.reset()
        self._pointer_y_filter.reset()

    # -----------------------------------------------------------------
    # Swipe detection
    # -----------------------------------------------------------------

    def _update_motion_history(
        self,
        timestamp: float,
        palm_x: float,
        palm_y: float,
    ) -> None:
        self._motion_history.append((timestamp, palm_x, palm_y))

        cutoff = timestamp - self.config.swipe_history_seconds

        while self._motion_history and self._motion_history[0][0] < cutoff:
            self._motion_history.popleft()

    def _detect_swipe(
        self,
        packet: LandmarkPacket,
        features: HandFeatures,
    ) -> Optional[GestureEvent]:
        if len(self._motion_history) < 3:
            return None

        start_t, start_x, start_y = self._motion_history[0]
        end_t, end_x, end_y = self._motion_history[-1]

        dt = max(end_t - start_t, 1e-6)
        dx = end_x - start_x
        dy = end_y - start_y

        if self.config.mirror_swipe_x:
            dx = -dx

        abs_dx = abs(dx)
        abs_dy = abs(dy)
        velocity = abs_dx / dt

        if abs_dx < self.config.swipe_min_dx:
            return None

        if abs_dy > self.config.swipe_max_dy:
            return None

        if velocity < self.config.swipe_min_velocity:
            return None

        gesture = NEXT_SLIDE if dx > 0 else PREVIOUS_SLIDE

        if not self._cooldown_ok(gesture, packet.timestamp, self.config.swipe_cooldown_seconds):
            return None

        displacement_score = self._score_range(
            abs_dx,
            self.config.swipe_min_dx,
            self.config.swipe_min_dx * 1.85,
        )

        velocity_score = self._score_range(
            velocity,
            self.config.swipe_min_velocity,
            self.config.swipe_min_velocity * 2.00,
        )

        vertical_penalty = 1.0 - self._score_range(
            abs_dy,
            self.config.swipe_max_dy * 0.50,
            self.config.swipe_max_dy,
        )

        # Base 0.50 floor so any swipe that passes the gate has meaningful
        # confidence. Without this, a swipe that barely clears the minimum
        # displacement/velocity scores ~0.15 and gets filtered by the UI.
        confidence = self._clamp01(
            0.50
            + 0.20 * displacement_score
            + 0.20 * velocity_score
            + 0.10 * vertical_penalty
        )

        self._last_emitted_at[gesture] = packet.timestamp
        self._motion_history.clear()

        return GestureEvent(
            gesture_type=gesture,
            confidence=confidence,
            timestamp=packet.timestamp,
            source_frame_id=packet.frame_id,
            metadata={
                "reason": "horizontal palm swipe",
                "dx": dx,
                "dy": dy,
                "velocity": velocity,
                "palm_x": features.palm_x,
                "palm_y": features.palm_y,
            },
        )

    # -----------------------------------------------------------------
    # Pinch zoom detection
    # -----------------------------------------------------------------

    def _detect_pinch_transition(
        self,
        packet: LandmarkPacket,
        features: HandFeatures,
    ) -> Optional[GestureEvent]:
        """Step-based pinch zoom.

        Track the thumb-index distance. Each time it shifts by
        pinch_step_distance from the last baseline, fire a zoom event in
        the direction of the shift. After firing, opposite-direction events
        are suppressed for a short window so the user can reset their hand
        position (e.g., close fingers to spread again for another zoom in)
        without accidentally triggering the reverse zoom.
        """
        fold = features.finger_folded
        ext = features.finger_extended

        other_fingers_folded = self._mean(
            fold["middle"],
            fold["ring"],
            fold["pinky"],
        )

        # Stricter than before: require index to be EXTENDED (not just
        # "not folded"). Prevents thumbs-up and similar poses from passing
        # the pinch-shape gate.
        thumb_index_active = self._mean(
            max(features.thumb_open_score, features.thumb_up_score),
            ext["index"],
        )

        pinch_shape_score = self._mean(
            other_fingers_folded,
            thumb_index_active,
        )

        # Not in pinch shape → clear baseline; no zoom this frame.
        if pinch_shape_score < self.config.pinch_shape_min_score:
            self._pinch_baseline_distance = None
            return None

        current_distance = features.pinch_distance

        # First frame of a fresh pinch session — establish baseline only.
        if self._pinch_baseline_distance is None:
            self._pinch_baseline_distance = current_distance
            return None

        delta = current_distance - self._pinch_baseline_distance

        # Haven't moved enough since last fire/baseline; wait for more.
        if abs(delta) < self.config.pinch_step_distance:
            return None

        gesture = ZOOM_IN if delta > 0 else ZOOM_OUT

        # Suppress reverse-direction firing while the user is "resetting"
        # their hand after a recent zoom. Refresh the suppression window
        # on every reverse-motion frame so a slow reset can't slip through.
        is_reverse = (
            self._last_pinch_gesture is not None
            and gesture != self._last_pinch_gesture
        )
        if is_reverse and packet.timestamp < self._pinch_reverse_until:
            self._pinch_baseline_distance = current_distance
            self._pinch_reverse_until = (
                packet.timestamp
                + self.config.pinch_reverse_suppression_seconds
            )
            return None

        if not self._cooldown_ok(
            gesture,
            packet.timestamp,
            self.config.pinch_cooldown_seconds,
        ):
            return None

        # Commit the fire.
        self._last_emitted_at[gesture] = packet.timestamp
        self._last_pinch_gesture = gesture
        self._pinch_reverse_until = (
            packet.timestamp + self.config.pinch_reverse_suppression_seconds
        )
        self._pinch_baseline_distance = current_distance

        step_score = self._score_range(
            abs(delta),
            self.config.pinch_step_distance,
            self.config.pinch_step_distance * 2.0,
        )
        confidence = self._clamp01(
            0.55
            + 0.25 * step_score
            + 0.20 * pinch_shape_score
        )

        return GestureEvent(
            gesture_type=gesture,
            confidence=confidence,
            timestamp=packet.timestamp,
            source_frame_id=packet.frame_id,
            metadata={
                "reason": "pinch step",
                "delta": delta,
                "current_distance": current_distance,
                "pinch_shape_score": pinch_shape_score,
            },
        )

    # -----------------------------------------------------------------
    # Cursor smoothing
    # -----------------------------------------------------------------

    def _smooth_cursor(
        self,
        x: Optional[float],
        y: Optional[float],
        timestamp: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        if x is None or y is None:
            return None, None

        sx = self._pointer_x_filter.filter(float(x), timestamp)
        sy = self._pointer_y_filter.filter(float(y), timestamp)

        return self._clamp01(sx), self._clamp01(sy)

    # -----------------------------------------------------------------
    # Event output
    # -----------------------------------------------------------------

    def _emit(self, event: GestureEvent) -> None:
        if self.on_gesture is None:
            return

        payload: Union[GestureEvent, Dict[str, Any]]

        if self.config.emit_dict_to_callback:
            payload = event.to_dict()
        else:
            payload = event

        self.on_gesture(payload)

    # -----------------------------------------------------------------
    # Packet coercion
    # -----------------------------------------------------------------

    def _coerce_landmark_packet(
        self,
        packet: Union[LandmarkPacket, Mapping[str, Any]],
    ) -> LandmarkPacket:
        if isinstance(packet, LandmarkPacket):
            return packet

        landmarks_raw = (
            packet.get("landmarks")
            or packet.get("hand_landmarks")
            or packet.get("points")
        )

        if landmarks_raw is None:
            raise ValueError(
                "Landmark packet must contain 'landmarks', "
                "'hand_landmarks', or 'points'."
            )

        landmarks = [self._coerce_landmark_point(p) for p in landmarks_raw]

        return LandmarkPacket(
            frame_id=int(packet.get("frame_id", packet.get("id", 0))),
            timestamp=float(packet.get("timestamp", time.time())),
            landmarks=landmarks,
            handedness=packet.get("handedness"),
            hand_score=packet.get("hand_score", packet.get("score")),
            frame_width=packet.get("frame_width", packet.get("width")),
            frame_height=packet.get("frame_height", packet.get("height")),
            metadata=dict(packet.get("metadata", {})),
        )

    @staticmethod
    def _coerce_landmark_point(point: Any) -> LandmarkPoint:
        if isinstance(point, LandmarkPoint):
            return point

        if isinstance(point, Mapping):
            return LandmarkPoint(
                x=float(point["x"]),
                y=float(point["y"]),
                z=float(point.get("z", 0.0)),
                visibility=point.get("visibility"),
                presence=point.get("presence"),
            )

        if isinstance(point, (tuple, list)):
            if len(point) < 2:
                raise ValueError("Landmark tuple/list must contain at least x and y.")

            return LandmarkPoint(
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]) if len(point) >= 3 else 0.0,
            )

        # Supports MediaPipe NormalizedLandmark-like objects with .x/.y/.z.
        if hasattr(point, "x") and hasattr(point, "y"):
            return LandmarkPoint(
                x=float(point.x),
                y=float(point.y),
                z=float(getattr(point, "z", 0.0)),
                visibility=getattr(point, "visibility", None),
                presence=getattr(point, "presence", None),
            )

        raise TypeError(f"Unsupported landmark point type: {type(point)!r}")

    # -----------------------------------------------------------------
    # Math helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _distance(a: LandmarkPoint, b: LandmarkPoint) -> float:
        return math.sqrt(
            (a.x - b.x) ** 2
            + (a.y - b.y) ** 2
            + (a.z - b.z) ** 2
        )

    @staticmethod
    def _angle_degrees(
        a: LandmarkPoint,
        b: LandmarkPoint,
        c: LandmarkPoint,
    ) -> float:
        """
        Returns angle ABC in degrees.
        """

        bax = a.x - b.x
        bay = a.y - b.y
        baz = a.z - b.z

        bcx = c.x - b.x
        bcy = c.y - b.y
        bcz = c.z - b.z

        dot = bax * bcx + bay * bcy + baz * bcz

        mag_ba = math.sqrt(bax * bax + bay * bay + baz * baz)
        mag_bc = math.sqrt(bcx * bcx + bcy * bcy + bcz * bcz)

        denom = max(mag_ba * mag_bc, 1e-9)
        cos_angle = max(-1.0, min(1.0, dot / denom))

        return math.degrees(math.acos(cos_angle))

    @staticmethod
    def _score_range(value: float, low: float, high: float) -> float:
        if high <= low:
            return 1.0 if value >= high else 0.0

        return max(0.0, min(1.0, (value - low) / (high - low)))

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _mean(*values: float) -> float:
        if not values:
            return 0.0

        return sum(values) / len(values)

    def _cooldown_ok(
        self,
        gesture_type: str,
        timestamp: float,
        override_cooldown: Optional[float] = None,
    ) -> bool:
        if override_cooldown is not None:
            cooldown = override_cooldown
        elif gesture_type in self.config.discrete_static_cooldowns:
            cooldown = self.config.discrete_static_cooldowns[gesture_type]
        else:
            cooldown = 0.0

        last = self._last_emitted_at.get(gesture_type, -1e9)
        return timestamp - last >= cooldown