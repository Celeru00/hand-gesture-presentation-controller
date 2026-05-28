"""
vision_engine.py
================
Dev 1 — Computer Vision Engine

Responsibilities:
- Open webcam or video file as requested by Dev 3 via SourceRequest.
- Run MediaPipe Hands on each frame to detect hand landmarks.
- Draw hand landmarks on the frame for visual feedback.
- Send annotated frames to the UI via app.receive_frame().

Does NOT:
- Classify gestures.
- Control the UI layout or widgets.
- Send keyboard or mouse events.

Integration:
    Dev 3 passes vision.handle_source_request as the on_source_selected
    callback when constructing PresentationControllerApp.

    Dev 1 calls app.receive_frame(packet) every frame.

Dev 2 hook:
    When Dev 2 is ready, set engine.gesture_engine = Dev2_instance.
    VisionEngine will call gesture_engine.receive_landmarks(landmarks)
    each frame once that attribute is set.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional, Union

import cv2
import mediapipe as mp

from contracts import FramePacket, SourceRequest


class VisionEngine:
    """
    Captures frames from webcam or video file, runs MediaPipe hand
    detection, and pushes annotated FramePacket objects to the UI.
    """

    def __init__(self, app: Any) -> None:
        """
        Parameters
        ----------
        app:
            The PresentationControllerApp instance (Dev 3).
            VisionEngine calls app.receive_frame() each frame.
        """
        self._app = app
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_id = 0

        # Dev 2 connection point.
        # Set this to the gesture engine instance when Dev 2 is ready.
        # VisionEngine will then forward landmark data each frame.
        self.gesture_engine: Optional[Any] = None

        # MediaPipe setup
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_draw_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

    # ------------------------------------------------------------------
    # Source control — called by Dev 3 via on_source_selected callback
    # ------------------------------------------------------------------

    def handle_source_request(self, request: SourceRequest) -> None:
        """
        Dev 3 calls this whenever the user changes the input source.
        Stops any running capture thread before starting a new one.
        """
        if request.source_type == "stop":
            self.stop()
        elif request.source_type == "webcam":
            self.stop()
            index = int(request.value) if request.value is not None else 0
            self._start(index)
        elif request.source_type == "video_file":
            self.stop()
            self._start(str(request.value))

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._stop_event.clear()
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal — capture thread
    # ------------------------------------------------------------------

    def _start(self, source: Union[int, str]) -> None:
        self._thread = threading.Thread(
            target=self._capture_loop,
            args=(source,),
            daemon=True,
            name="VisionEngine",
        )
        self._thread.start()

    def _capture_loop(self, source: Union[int, str]) -> None:
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            self._app.last_action_var.set(
                f"Could not open source: {source}"
            )
            return

        target_fps = 30
        frame_interval = 1.0 / target_fps
        is_file = isinstance(source, str)

        try:
            while not self._stop_event.is_set():
                t_start = time.perf_counter()

                ret, frame = cap.read()

                if not ret:
                    if is_file:
                        # Loop video file back to the beginning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                annotated, landmarks = self._process_frame(frame)

                # Forward raw landmark results to Dev 2 if connected
                if self.gesture_engine is not None and landmarks is not None:
                    try:
                        self.gesture_engine.receive_landmarks(
                            landmarks=landmarks,
                            frame_id=self._frame_id,
                            width=frame.shape[1],
                            height=frame.shape[0],
                        )
                    except Exception:
                        pass  # Dev 2 not ready yet — fail silently

                # Send annotated frame to Dev 3's UI
                packet = FramePacket(
                    frame_id=self._frame_id,
                    timestamp=time.time(),
                    image=annotated,
                    color_format="BGR",
                    width=annotated.shape[1],
                    height=annotated.shape[0],
                )
                self._app.receive_frame(packet)
                self._frame_id += 1

                # Sleep to maintain target FPS
                elapsed = time.perf_counter() - t_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            cap.release()

    def _process_frame(self, frame):
        """
        Run MediaPipe on one frame.

        Returns
        -------
        annotated : np.ndarray
            BGR frame with hand landmarks drawn on it.
        landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList or None
            Raw MediaPipe landmarks for Dev 2. None if no hand detected.
        """
        # MediaPipe requires RGB, and writeable=False is a performance hint
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()
        landmarks = None

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]

            self._mp_draw.draw_landmarks(
                annotated,
                landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_draw_styles.get_default_hand_landmarks_style(),
                self._mp_draw_styles.get_default_hand_connections_style(),
            )

        return annotated, landmarks
