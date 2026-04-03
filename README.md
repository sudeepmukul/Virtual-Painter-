# Virtual Painter 

A real-time hand tracking drawing app using MediaPipe and OpenCV.

Control a virtual brush using your fingers through your webcam.

---

##  Features

* Draw in air using your index finger
* Smooth strokes (no jitter)
* Adjustable brush thickness
* Color selection using on-screen bar
* Eraser mode
* Full hand tracking overlay (for visualization)

---

##  Tech Stack

* Python
* OpenCV
* MediaPipe (Tasks API)

---

##  Controls

| Gesture         | Action                 |
| --------------- | ---------------------- |
| Index finger up | Draw                   |
| Index + Middle  | Select color (top bar) |
| Thumb + Index   | Adjust brush thickness |
| Fist            | Clear canvas           |

---

##  Setup

### 1. Clone the repo

```bash
git clone https://github.com/sudeepmukul/Virtual-Painter-.git
cd Virtual-Painter-
```

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

### 4. Download model file

Download this file and place it in the project folder:

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

---

##  Run

```bash
python Virtual_Painter.py
```

---

##  Notes

* Works best with good lighting
* Keep hand fully visible for stable tracking
* If tracking feels shaky, move slower

---

##  Future Improvements

* Save drawings
* Undo/redo gestures
* Better UI (floating palette)
* Multi-hand support

---

## 👤 Author

Made by Sudeep <3
