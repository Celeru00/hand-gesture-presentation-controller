"""
main.py
=======
Pipeline entry point for the Hand Gesture Presentation Controller.

Wires Dev 1 (VisionEngine) and Dev 3 (PresentationControllerApp) together.
Dev 2 (GestureEngine) slot is intentionally left as a comment — to be added
once Dev 2's module is ready.

Run with:
    uv run main.py
"""

from control_panel import PresentationControllerApp
from vision_engine import VisionEngine


def main() -> None:
    # Dev 3 — UI + Controller
    app = PresentationControllerApp()

    # Dev 1 — Vision Engine
    vision = VisionEngine(app)

    # Wire Dev 3 → Dev 1:
    # When the user clicks "Use webcam" or "Choose video file" in the UI,
    # Dev 3 calls app.on_source_selected(request), which forwards to
    # vision.handle_source_request(request).
    app.on_source_selected = vision.handle_source_request

    # Dev 2 — Gesture Engine (not yet implemented)
    # Once Dev 2's module exists, add:
    #
    #   from gesture_engine import GestureEngine
    #   gesture = GestureEngine(app)
    #   vision.gesture_engine = gesture
    #
    # Dev 2 receives raw landmarks from vision.gesture_engine.receive_landmarks()
    # and sends confirmed gestures via app.receive_gesture(event).

    app.run()


if __name__ == "__main__":
    main()
