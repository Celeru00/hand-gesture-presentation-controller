"""
control_panel.py
=============
Tkinter UI + PowerPoint Controller

Responsibilities:
- Show a desktop UI.
- Let the user request webcam or video-file input.
- Display live frames supplied by vision_engine.py.
- Display stable gesture events supplied by gesture_engine.py.
- Translate confirmed gesture events into PowerPoint keyboard/mouse actions.

This file does NOT:
- Open a webcam.
- Open a video file.
- Run MediaPipe.
- Classify gestures.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Union

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from contracts import FramePacket, GestureEvent, SourceRequest

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise RuntimeError("Pillow is required. Install it with: pip install pillow") from exc

try:
    import pyautogui
except ImportError:
    pyautogui = None  # type: ignore


# Canonical gesture names
NEXT_SLIDE = "next_slide"
PREVIOUS_SLIDE = "previous_slide"
START_PRESENTATION = "start_presentation"
STOP_EXIT = "stop_exit"
BLANK_SCREEN = "blank_screen"
LASER_POINTER = "laser_pointer"
DRAW_ANNOTATE = "draw_annotate"
ZOOM_IN = "zoom_in"
ZOOM_OUT = "zoom_out"

POINTER_GESTURES = {LASER_POINTER, DRAW_ANNOTATE}

GESTURE_LABELS = {
    NEXT_SLIDE: "Next slide",
    PREVIOUS_SLIDE: "Previous slide",
    START_PRESENTATION: "Start presentation",
    STOP_EXIT: "Stop / exit",
    BLANK_SCREEN: "Blank screen",
    LASER_POINTER: "Laser pointer",
    DRAW_ANNOTATE: "Draw / annotate",
    ZOOM_IN: "Zoom in",
    ZOOM_OUT: "Zoom out",
}


@dataclass
class ControllerConfig:
    min_confidence: float = 0.75
    enable_actions_on_start: bool = True

    frame_queue_size: int = 2
    event_queue_size: int = 64
    ui_poll_ms: int = 15

    # Pointer smoothing.
    # 1.0 = no smoothing.
    # 0.2 = smoother but slower.
    pointer_new_position_weight: float = 0.45

    # Set True if your camera feed is mirrored and pointer movement feels reversed.
    mirror_cursor_x: bool = False

    # Prevents the mouse from staying held down forever in draw mode.
    draw_release_timeout_seconds: float = 0.25

    # Optional. If installed:
    #   pip install pygetwindow
    # You can set target_window_title_keyword="PowerPoint"
    # and try_focus_target_before_action=True.
    target_window_title_keyword: Optional[str] = None
    try_focus_target_before_action: bool = False

    action_cooldowns: Dict[str, float] = field(
        default_factory=lambda: {
            NEXT_SLIDE: 0.80,
            PREVIOUS_SLIDE: 0.80,
            START_PRESENTATION: 2.00,
            STOP_EXIT: 1.50,
            BLANK_SCREEN: 1.00,
            ZOOM_IN: 0.25,
            ZOOM_OUT: 0.25,
            LASER_POINTER: 0.00,
            DRAW_ANNOTATE: 0.00,
        }
    )


def normalize_gesture_name(name: str) -> str:
    """
    Allows Dev 2 to send either:
        "Next slide"
        "next_slide"
        "swipe_right"
    etc.
    """

    if not name:
        return ""

    key = name.strip().lower()
    key = key.replace("/", " ")
    key = key.replace("-", " ")
    key = key.replace("+", " plus ")
    key = "_".join(key.split())

    aliases = {
        "next": NEXT_SLIDE,
        "next_slide": NEXT_SLIDE,
        "swipe_right": NEXT_SLIDE,
        "right_swipe": NEXT_SLIDE,

        "previous": PREVIOUS_SLIDE,
        "prev": PREVIOUS_SLIDE,
        "previous_slide": PREVIOUS_SLIDE,
        "prev_slide": PREVIOUS_SLIDE,
        "swipe_left": PREVIOUS_SLIDE,
        "left_swipe": PREVIOUS_SLIDE,

        "start": START_PRESENTATION,
        "start_presentation": START_PRESENTATION,
        "open_palm": START_PRESENTATION,
        "open_hand": START_PRESENTATION,
        "palm": START_PRESENTATION,

        "stop": STOP_EXIT,
        "exit": STOP_EXIT,
        "stop_exit": STOP_EXIT,
        "closed_fist": STOP_EXIT,
        "fist": STOP_EXIT,

        "blank": BLANK_SCREEN,
        "blank_screen": BLANK_SCREEN,
        "thumbs_up": BLANK_SCREEN,
        "thumb_up": BLANK_SCREEN,

        "laser": LASER_POINTER,
        "laser_pointer": LASER_POINTER,
        "index_finger": LASER_POINTER,
        "index_finger_only": LASER_POINTER,
        "index_only": LASER_POINTER,
        "pointing_up": LASER_POINTER,

        "draw": DRAW_ANNOTATE,
        "annotate": DRAW_ANNOTATE,
        "draw_annotate": DRAW_ANNOTATE,
        "peace": DRAW_ANNOTATE,
        "peace_sign": DRAW_ANNOTATE,
        "two_fingers": DRAW_ANNOTATE,

        "zoom_in": ZOOM_IN,
        "pinch_open": ZOOM_IN,
        "pinch_fingers_open": ZOOM_IN,

        "zoom_out": ZOOM_OUT,
        "pinch_closed": ZOOM_OUT,
        "pinch_fingers_closed": ZOOM_OUT,
    }

    return aliases.get(key, key)


def _put_latest(item_queue: queue.Queue, item: Any) -> None:
    """
    Put item in a bounded queue.

    If the queue is full, the oldest item is dropped.
    This keeps the UI responsive and avoids laggy video.
    """

    try:
        item_queue.put_nowait(item)
    except queue.Full:
        try:
            item_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            item_queue.put_nowait(item)
        except queue.Full:
            pass


class PowerPointActionDriver:
    """
    Converts stable GestureEvent objects into pyautogui actions.
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self._last_action_at: Dict[str, float] = {}
        self._mouse_is_down = False
        self._last_draw_event_at = 0.0
        self._last_pointer_xy: Optional[tuple[int, int]] = None

        if pyautogui is not None:
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.02

    def execute(
        self,
        event: GestureEvent,
        control_enabled: bool,
        confidence_threshold: float,
    ) -> str:
        gesture = normalize_gesture_name(event.gesture_type)
        label = GESTURE_LABELS.get(gesture, event.gesture_type)

        if event.confidence < confidence_threshold:
            return f"Ignored {label}: confidence {event.confidence:.2f} below threshold."

        if gesture not in GESTURE_LABELS:
            return f"Ignored unknown gesture: {event.gesture_type!r}."

        if not control_enabled:
            return f"Detected {label}; PowerPoint control is disabled."

        if pyautogui is None:
            return "pyautogui is not installed; action was not sent."

        if self.config.try_focus_target_before_action:
            self._try_focus_target_window()

        if gesture in POINTER_GESTURES:
            return self._handle_pointer_gesture(gesture, event)

        # Any discrete action stops drawing first.
        self.release_draw_if_needed(force=True)

        now = time.time()
        cooldown = self.config.action_cooldowns.get(gesture, 0.0)
        last = self._last_action_at.get(gesture, 0.0)

        if now - last < cooldown:
            return f"Cooldown active for {label}; action not repeated."

        self._last_action_at[gesture] = now

        if gesture == NEXT_SLIDE:
            pyautogui.press("right")
            return "Sent Right Arrow: next slide."

        if gesture == PREVIOUS_SLIDE:
            pyautogui.press("left")
            return "Sent Left Arrow: previous slide."

        if gesture == START_PRESENTATION:
            pyautogui.press("f5")
            return "Sent F5: start presentation."

        if gesture == STOP_EXIT:
            pyautogui.press("esc")
            return "Sent Escape: exit slideshow."

        if gesture == BLANK_SCREEN:
            pyautogui.press("b")
            return "Sent B: blank/unblank screen."

        if gesture == ZOOM_IN:
            pyautogui.hotkey("ctrl", "+")
            return "Sent Ctrl +: zoom in."

        if gesture == ZOOM_OUT:
            pyautogui.hotkey("ctrl", "-")
            return "Sent Ctrl -: zoom out."

        return f"No action configured for {label}."

    def release_draw_if_needed(self, force: bool = False) -> None:
        if pyautogui is None:
            self._mouse_is_down = False
            return

        expired = (
            self._mouse_is_down
            and time.time() - self._last_draw_event_at
            > self.config.draw_release_timeout_seconds
        )

        if self._mouse_is_down and (force or expired):
            pyautogui.mouseUp()
            self._mouse_is_down = False

    def _handle_pointer_gesture(self, gesture: str, event: GestureEvent) -> str:
        coords = self._get_event_cursor(event)
        label = GESTURE_LABELS[gesture]

        if coords is None:
            if gesture == DRAW_ANNOTATE:
                self.release_draw_if_needed(force=True)
            return f"Detected {label}, but no cursor_x/cursor_y was supplied."

        x, y = self._to_screen_xy(coords[0], coords[1])

        if gesture == LASER_POINTER:
            self.release_draw_if_needed(force=True)
            pyautogui.moveTo(x, y, duration=0)
            return f"Moved pointer to ({x}, {y})."

        if gesture == DRAW_ANNOTATE:
            if not self._mouse_is_down:
                pyautogui.moveTo(x, y, duration=0)
                pyautogui.mouseDown()
                self._mouse_is_down = True
            else:
                pyautogui.moveTo(x, y, duration=0)

            self._last_draw_event_at = time.time()
            return f"Drawing at ({x}, {y})."

        return f"No pointer action configured for {label}."

    def _get_event_cursor(self, event: GestureEvent) -> Optional[tuple[float, float]]:
        x = event.cursor_x
        y = event.cursor_y

        if x is None:
            x = event.metadata.get("cursor_x", event.metadata.get("x"))
        if y is None:
            y = event.metadata.get("cursor_y", event.metadata.get("y"))

        if x is None or y is None:
            return None

        try:
            return float(x), float(y)
        except (TypeError, ValueError):
            return None

    def _to_screen_xy(self, x: float, y: float) -> tuple[int, int]:
        screen_w, screen_h = pyautogui.size()

        normalized = 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

        if normalized:
            if self.config.mirror_cursor_x:
                x = 1.0 - x

            sx = int(x * screen_w)
            sy = int(y * screen_h)
        else:
            sx = int(x)
            sy = int(y)

        sx = max(0, min(screen_w - 1, sx))
        sy = max(0, min(screen_h - 1, sy))

        weight = max(0.0, min(1.0, self.config.pointer_new_position_weight))

        if self._last_pointer_xy is not None and weight < 1.0:
            old_x, old_y = self._last_pointer_xy
            sx = int(old_x * (1.0 - weight) + sx * weight)
            sy = int(old_y * (1.0 - weight) + sy * weight)

        self._last_pointer_xy = (sx, sy)
        return sx, sy

    def _try_focus_target_window(self) -> None:
        keyword = self.config.target_window_title_keyword

        if not keyword:
            return

        try:
            import pygetwindow as gw
        except Exception:
            return

        try:
            windows = gw.getWindowsWithTitle(keyword)

            for window in windows:
                if getattr(window, "isMinimized", False):
                    continue

                window.activate()
                time.sleep(0.05)
                return
        except Exception:
            return


class PresentationControllerApp:
    """
    Tkinter UI for Dev 3.

    Thread-safe public methods:
        receive_frame(packet)
        receive_gesture(event)

    Dev 1 calls receive_frame(...) each frame.
    Dev 2 calls receive_gesture(...) on confirmed gestures.
    """

    def __init__(
        self,
        root: Optional[tk.Tk] = None,
        on_source_selected: Optional[Callable[[SourceRequest], None]] = None,
        config: Optional[ControllerConfig] = None,
    ):
        self.root = root or tk.Tk()
        self.config = config or ControllerConfig()
        self.on_source_selected = on_source_selected or (lambda request: None)

        self.frame_queue: queue.Queue = queue.Queue(maxsize=self.config.frame_queue_size)
        self.event_queue: queue.Queue = queue.Queue(maxsize=self.config.event_queue_size)

        self.driver = PowerPointActionDriver(self.config)

        self._closed = False
        self._latest_photo: Optional[ImageTk.PhotoImage] = None

        self.webcam_index_var = tk.IntVar(value=0)
        self.control_enabled_var = tk.BooleanVar(value=self.config.enable_actions_on_start)
        self.confidence_threshold_var = tk.DoubleVar(value=self.config.min_confidence)
        self.confidence_percent_var = tk.DoubleVar(value=0.0)

        self.source_status_var = tk.StringVar(value="No source selected.")
        self.current_gesture_var = tk.StringVar(value="None")
        self.current_confidence_var = tk.StringVar(value="0.00")
        self.last_action_var = tk.StringVar(value="No action sent yet.")
        self.threshold_label_var = tk.StringVar(value=f"{self.config.min_confidence:.2f}")

        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(self.config.ui_poll_ms, self._poll_queues)

    # ------------------------------------------------------------------
    # Public API called by Dev 1 and Dev 2
    # ------------------------------------------------------------------

    def receive_frame(self, packet: Union[FramePacket, Mapping[str, Any]]) -> None:
        """
        Thread-safe. Dev 1 calls this whenever a new frame is ready.
        """

        try:
            frame_packet = self._coerce_frame_packet(packet)
        except Exception as exc:
            self.last_action_var.set(f"Bad frame packet: {exc}")
            return

        _put_latest(self.frame_queue, frame_packet)

    def receive_gesture(self, event: Union[GestureEvent, Mapping[str, Any]]) -> None:
        """
        Thread-safe. Dev 2 calls this only for stable confirmed gestures.
        """

        try:
            gesture_event = self._coerce_gesture_event(event)
        except Exception as exc:
            self.last_action_var.set(f"Bad gesture event: {exc}")
            return

        _put_latest(self.event_queue, gesture_event)

    # Friendly aliases
    push_frame = receive_frame
    push_gesture_event = receive_gesture

    def run(self) -> None:
        self.root.mainloop()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.title("Hand Gesture Presentation Controller")
        self.root.minsize(900, 650)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        self._build_source_panel(main)
        self._build_video_panel(main)
        self._build_status_panel(main)

    def _build_source_panel(self, parent: ttk.Frame) -> None:
        source = ttk.LabelFrame(parent, text="Input source request")
        source.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        source.columnconfigure(6, weight=1)

        ttk.Label(source, text="Webcam index:").grid(
            row=0, column=0, padx=(8, 4), pady=8,
        )

        tk.Spinbox(
            source, from_=0, to=10, width=5, textvariable=self.webcam_index_var,
        ).grid(row=0, column=1, padx=4, pady=8)

        ttk.Button(
            source, text="Use webcam", command=self._request_webcam,
        ).grid(row=0, column=2, padx=4, pady=8)

        ttk.Button(
            source, text="Choose video file", command=self._request_video_file,
        ).grid(row=0, column=3, padx=4, pady=8)

        ttk.Button(
            source, text="Stop input", command=self._request_stop_input,
        ).grid(row=0, column=4, padx=4, pady=8)

        ttk.Label(source, textvariable=self.source_status_var).grid(
            row=0, column=5, columnspan=2, padx=(16, 8), pady=8, sticky="w",
        )

    def _build_video_panel(self, parent: ttk.Frame) -> None:
        video = ttk.LabelFrame(parent, text="Live feed from Dev 1")
        video.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        video.columnconfigure(0, weight=1)
        video.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(
            video,
            text="No frame received yet.",
            anchor="center",
            background="black",
            foreground="white",
        )
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _build_status_panel(self, parent: ttk.Frame) -> None:
        status = ttk.LabelFrame(parent, text="Gesture and PowerPoint control")
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(1, weight=1)
        status.columnconfigure(3, weight=1)

        ttk.Label(status, text="Current gesture:").grid(
            row=0, column=0, padx=(8, 4), pady=6, sticky="w",
        )
        ttk.Label(status, textvariable=self.current_gesture_var).grid(
            row=0, column=1, padx=4, pady=6, sticky="w",
        )
        ttk.Label(status, text="Confidence:").grid(
            row=0, column=2, padx=(8, 4), pady=6, sticky="w",
        )
        ttk.Label(status, textvariable=self.current_confidence_var).grid(
            row=0, column=3, padx=4, pady=6, sticky="w",
        )

        ttk.Progressbar(
            status, variable=self.confidence_percent_var, maximum=100.0,
        ).grid(row=1, column=0, columnspan=4, padx=8, pady=(0, 8), sticky="ew")

        ttk.Label(status, text="Min confidence:").grid(
            row=2, column=0, padx=(8, 4), pady=6, sticky="w",
        )
        ttk.Scale(
            status,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.confidence_threshold_var,
            command=self._on_threshold_changed,
        ).grid(row=2, column=1, padx=4, pady=6, sticky="ew")
        ttk.Label(status, textvariable=self.threshold_label_var, width=5).grid(
            row=2, column=2, padx=4, pady=6, sticky="w",
        )
        ttk.Checkbutton(
            status,
            text="Enable PowerPoint control",
            variable=self.control_enabled_var,
        ).grid(row=2, column=3, padx=8, pady=6, sticky="w")

        ttk.Label(status, text="Last action:").grid(
            row=3, column=0, padx=(8, 4), pady=6, sticky="w",
        )
        ttk.Label(status, textvariable=self.last_action_var).grid(
            row=3, column=1, columnspan=3, padx=4, pady=6, sticky="w",
        )

        ttk.Button(
            status,
            text="Release mouse",
            command=lambda: self.driver.release_draw_if_needed(force=True),
        ).grid(row=4, column=0, padx=8, pady=(4, 8), sticky="w")
        ttk.Button(
            status, text="Clear status", command=self._clear_status,
        ).grid(row=4, column=1, padx=4, pady=(4, 8), sticky="w")

    # ------------------------------------------------------------------
    # Source selection: Dev 3 -> Dev 1
    # ------------------------------------------------------------------

    def _request_webcam(self) -> None:
        index = int(self.webcam_index_var.get())
        request = SourceRequest(source_type="webcam", value=index)
        self.source_status_var.set(f"Requested webcam index {index}.")
        self._send_source_request(request)

    def _request_video_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose video file",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        request = SourceRequest(source_type="video_file", value=path)
        self.source_status_var.set(f"Requested video file: {path}")
        self._send_source_request(request)

    def _request_stop_input(self) -> None:
        request = SourceRequest(source_type="stop", value=None)
        self.source_status_var.set("Requested input stop.")
        self._send_source_request(request)

    def _send_source_request(self, request: SourceRequest) -> None:
        try:
            self.on_source_selected(request)
        except Exception as exc:
            messagebox.showerror(
                "Source request failed",
                f"Could not send source request to Dev 1:\n\n{exc}",
            )

    # ------------------------------------------------------------------
    # Queue polling
    # ------------------------------------------------------------------

    def _poll_queues(self) -> None:
        if self._closed:
            return

        latest_frame = None
        while True:
            try:
                latest_frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break

        if latest_frame is not None:
            self._display_frame(latest_frame)

        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break
            self._process_gesture_event(event)

        self.driver.release_draw_if_needed(force=False)
        self.root.after(self.config.ui_poll_ms, self._poll_queues)

    def _process_gesture_event(self, event: GestureEvent) -> None:
        normalized = normalize_gesture_name(event.gesture_type)
        label = GESTURE_LABELS.get(normalized, event.gesture_type)

        self.current_gesture_var.set(label)
        self.current_confidence_var.set(f"{event.confidence:.2f}")
        self.confidence_percent_var.set(
            max(0.0, min(100.0, event.confidence * 100.0))
        )

        result = self.driver.execute(
            event=event,
            control_enabled=bool(self.control_enabled_var.get()),
            confidence_threshold=float(self.confidence_threshold_var.get()),
        )

        self.last_action_var.set(result)

    def _display_frame(self, packet: FramePacket) -> None:
        try:
            pil_image = self._packet_to_pil_image(packet)
        except Exception as exc:
            self.last_action_var.set(f"Could not display frame: {exc}")
            return

        label_w = max(320, self.video_label.winfo_width())
        label_h = max(240, self.video_label.winfo_height())

        display_image = pil_image.copy()

        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS  # type: ignore

        display_image.thumbnail((label_w, label_h), resample)

        self._latest_photo = ImageTk.PhotoImage(display_image)
        self.video_label.configure(image=self._latest_photo, text="")

    def _packet_to_pil_image(self, packet: FramePacket) -> Image.Image:
        image = packet.image

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if np is None:
            raise RuntimeError("numpy is required to display ndarray frames.")

        array = np.asarray(image)

        if array.ndim == 2:
            return Image.fromarray(self._as_uint8(array)).convert("RGB")

        if array.ndim != 3:
            raise ValueError(f"Expected HxW or HxWxC image, got shape {array.shape}.")

        color_format = (packet.color_format or "BGR").upper()
        array = self._as_uint8(array)

        if array.shape[2] == 3:
            if color_format == "BGR":
                array = array[:, :, ::-1]
            elif color_format != "RGB":
                raise ValueError(f"Unsupported 3-channel color_format: {color_format}")

            return Image.fromarray(np.ascontiguousarray(array), mode="RGB")

        if array.shape[2] == 4:
            if color_format == "BGRA":
                array = array[:, :, [2, 1, 0, 3]]
            elif color_format != "RGBA":
                raise ValueError(f"Unsupported 4-channel color_format: {color_format}")

            return Image.fromarray(np.ascontiguousarray(array), mode="RGBA").convert("RGB")

        raise ValueError(f"Unsupported channel count: {array.shape[2]}")

    @staticmethod
    def _as_uint8(array: Any) -> Any:
        if np is None:
            return array

        arr = np.asarray(array)

        if arr.dtype == np.uint8:
            return arr

        arr = arr.astype("float32")

        if arr.size and arr.max() <= 1.0:
            arr = arr * 255.0

        return np.clip(arr, 0, 255).astype("uint8")

    # ------------------------------------------------------------------
    # Coercion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_frame_packet(packet: Union[FramePacket, Mapping[str, Any]]) -> FramePacket:
        if isinstance(packet, FramePacket):
            return packet

        image = packet.get("image", packet.get("frame"))

        if image is None:
            raise ValueError("Frame packet must contain 'image' or 'frame'.")

        return FramePacket(
            frame_id=int(packet.get("frame_id", packet.get("id", 0))),
            timestamp=float(packet.get("timestamp", time.time())),
            image=image,
            color_format=str(packet.get("color_format", "BGR")),
            width=packet.get("width"),
            height=packet.get("height"),
        )

    @staticmethod
    def _coerce_gesture_event(
        event: Union[GestureEvent, Mapping[str, Any]]
    ) -> GestureEvent:
        if isinstance(event, GestureEvent):
            return event

        gesture_type = (
            event.get("gesture_type")
            or event.get("type")
            or event.get("gesture")
            or event.get("name")
        )

        if not gesture_type:
            raise ValueError(
                "Gesture event must contain 'gesture_type', 'type', or 'gesture'."
            )

        metadata = dict(event.get("metadata", {}))

        cursor_x = event.get("cursor_x", event.get("x", metadata.get("cursor_x")))
        cursor_y = event.get("cursor_y", event.get("y", metadata.get("cursor_y")))

        return GestureEvent(
            gesture_type=str(gesture_type),
            confidence=float(event.get("confidence", 1.0)),
            timestamp=float(event.get("timestamp", time.time())),
            cursor_x=None if cursor_x is None else float(cursor_x),
            cursor_y=None if cursor_y is None else float(cursor_y),
            source_frame_id=event.get("source_frame_id", event.get("frame_id")),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _on_threshold_changed(self, value: str) -> None:
        self.threshold_label_var.set(f"{float(value):.2f}")

    def _clear_status(self) -> None:
        self.current_gesture_var.set("None")
        self.current_confidence_var.set("0.00")
        self.confidence_percent_var.set(0.0)
        self.last_action_var.set("No action sent yet.")

    def _on_close(self) -> None:
        self._closed = True
        self.driver.release_draw_if_needed(force=True)

        try:
            self.on_source_selected(SourceRequest(source_type="stop", value=None))
        except Exception:
            pass

        self.root.destroy()
