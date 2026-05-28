# Hand Gesture Presentation Controller - CMSC 191 Final Project

**CMSC 191: Special Topics - Computer Vision with Python**
Section F-3L · A.Y. 2025-2026


## Members

| Name | Role |
|---|---|
| Janelle Ojanola | Computer Vision Engine |
| Renz Jaepril Mongaya | Gesture Recognition Engine |
| Francis Reid Arranguez | UI + Controller |


## Project Overview

This project is a real-time hand gesture recognition system that lets a user control **Microsoft PowerPoint presentations** using only hand gestures captured through a webcam or pre-recorded video file. No keyboard or mouse required during a presentation.

Gestures are detected using computer vision and translated into PowerPoint keyboard shortcuts — advancing slides, starting and stopping the slideshow, blanking the screen, and controlling an on-screen laser pointer and drawing tool.


## Pipeline

```
Webcam / Video File
        ↓
[ Vision Engine ]       detects hand, extracts 21 landmarks per frame
        ↓
[ Gesture Engine ]      classifies landmarks into a named gesture
        ↓
[ UI + Controller ]     displays live feed, fires keyboard shortcuts
        ↓
    PowerPoint
```

Each stage runs in its own thread. Stages communicate through a shared push interface using data types defined in `contracts.py`. No stage imports another stage's internal code.


## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Computer vision | OpenCV, MediaPipe Hands |
| Gesture control | PyAutoGUI |
| GUI | Tkinter (ttk) |
| Image processing | Pillow, NumPy |
| Package manager | uv |


## Computer Vision Approach

Hand detection and landmark extraction is handled by **MediaPipe Hands**, a pre-trained real-time hand tracking solution developed by Google. It detects a hand in the video frame and returns
21 landmarks — specific anatomical points on the hand such as fingertips, knuckle joints, and the wrist — each with normalized x, y, and z coordinates relative to the frame dimensions.

![System pipeline](https://mediapipe.dev/images/mobile/hand_landmarks.png)

```
MediaPipe hand landmark indices:

        8   12  16  20       ← fingertips
        |   |   |   |
        7   11  15  19
        |   |   |   |
        6   10  14  18
        |   |   |   |
    4   5    9  13  17
    |   |
    3   |
    |   |
    2   0  (wrist)
    |
    1
```

These 21 landmarks are forwarded to the gesture recognition layer, which applies rule-based geometric logic to classify the hand shape and motion into one of the supported gestures. No custom ML model is trained here. Classification relies on spatial relationships between landmark positions,
making it fast, interpretable, and lightweight.

## Supported Gestures

| Gesture | Trigger | PowerPoint Action |
|---|---|---|
| Next slide | Swipe hand right | → Arrow key |
| Previous slide | Swipe hand left | ← Arrow key |
| Start presentation | Open palm facing camera | F5 |
| Stop / exit | Closed fist | Escape |
| Blank screen | Thumbs up | B |
| Laser pointer | Index finger only pointing up | Move cursor |
| Draw / annotate | Peace sign (index + middle up) | Click and drag |
| Zoom in | Pinch open | Ctrl + |
| Zoom out | Pinch close | Ctrl - |


## File Structure

```
hand-gesture-presentation-controller/
├── shared.py                 shared data types for all three modules
├── vision_engine.py          webcam capture and landmark extraction
├── gesture_engine.py         gesture classification and smoothing
├── control_panel.py          Tkinter UI and PowerPoint control
├── main.py                   pipeline entry point
├── CMSC 191 Project.ipynb    notebook launcher for demo and submission
├── pyproject.toml            project metadata and dependencies
├── uv.lock                   locked dependency versions
├── .python-version           Python 3.11
└── README.md
```

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- A working webcam, or a `.mp4` / `.avi` / `.mov` video file as input
- Microsoft PowerPoint open and in focus when using gesture control


## Installation

```bash
# Clone the repository
git clone [repo url]
cd hand-gesture-presentation-controller

# Install all dependencies
# uv creates and manages the virtual environment automatically
uv sync
```


## How to Run

```bash
uv run main.py
```

The application window will open. Select your input source (webcam or video file) and click the appropriate button. Position your hand clearly in front of the camera. Switch to PowerPoint before performing gestures so that keyboard shortcuts are delivered to the correct window.

## Notes

- Single-hand detection only. If two hands are visible, only the first detected hand is used.
- Ensure adequate, even lighting for reliable landmark detection. Avoid strong backlight behind your hand.
- The confidence threshold can be adjusted in real time using the slider in the UI. Lower values are more sensitive but may cause false positives.
- PyAutoGUI's failsafe is enabled by default — move the mouse cursor to the very top-left corner of the screen to immediately abort any runaway keyboard or mouse action.
- If gesture control feels sluggish, check that PowerPoint is the active foreground window before performing gestures.
