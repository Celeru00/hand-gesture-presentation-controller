"""
main.py
=======
Pipeline entry point for the Hand Gesture Presentation Controller.

Full pipeline:
    Dev 1 (VisionEngine) → Dev 2 (GestureRecognitionEngine) → Dev 3 (PresentationControllerApp)

Run with:
    uv run main.py
"""

from engines.control_panel import PresentationControllerApp
from engines.gesture_engine import GestureRecognitionEngine
from engines.vision_engine import VisionEngine


def main() -> None:
    # Dev 3 — UI + Controller
    app = PresentationControllerApp()

    # Dev 2 — Gesture Engine
    # Receives LandmarkPackets from Dev 1, emits GestureEvents to Dev 3.
    gesture = GestureRecognitionEngine(on_gesture=app.receive_gesture)

    # Dev 1 — Vision Engine
    vision = VisionEngine(app)
    vision.gesture_engine = gesture

    # Wire Dev 3 → Dev 1: UI source selection drives the capture thread.
    app.on_source_selected = vision.handle_source_request

    app.run()


if __name__ == "__main__":
    main()
