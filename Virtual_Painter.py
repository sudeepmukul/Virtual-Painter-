import cv2
import numpy as np
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ------------------ CAMERA ------------------
cap = cv2.VideoCapture(0)
width, height = 1280, 720
cap.set(3, width)
cap.set(4, height)

# Canvas
imgCanvas = np.zeros((height, width, 3), np.uint8)

# Drawing settings
drawColor = (0, 0, 255)
thickness = 10
xp, yp = 0, 0

# Smoothing
smoothening = 0.7
prev_x, prev_y = 0, 0

# Gesture stability
gesture_buffer = []
BUFFER_SIZE = 5

# Colors
colors = [(0,0,255), (255,0,0), (0,255,0), (0,0,0)]
color_names = ["RED", "BLUE", "GREEN", "ERASER"]

# ------------------ MEDIAPIPE ------------------
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

detector = vision.HandLandmarker.create_from_options(options)

# ------------------ MAIN LOOP ------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    # 🎨 DRAW COLOR BAR
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (i*320, 0), ((i+1)*320, 100), color, cv2.FILLED)
        cv2.putText(frame, color_names[i], (i*320+40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            points = []

            # Convert landmarks
            for lm in hand_landmarks:
                x = int(lm.x * width)
                y = int(lm.y * height)
                points.append([x, y])

            if len(points) != 0:
                x1, y1 = points[8]   # index
                x2, y2 = points[12]  # middle
                x3, y3 = points[4]   # thumb

                # ------------------ SMOOTHING ------------------
                curr_x = int(prev_x * smoothening + x1 * (1 - smoothening))
                curr_y = int(prev_y * smoothening + y1 * (1 - smoothening))
                prev_x, prev_y = curr_x, curr_y

                # ------------------ FINGER DETECTION ------------------
                fingers = []

                fingers.append(1 if points[4][0] < points[3][0] else 0)

                tipIds = [8, 12, 16, 20]
                for id in tipIds:
                    fingers.append(1 if points[id][1] < points[id - 2][1] else 0)

                # ------------------ STABLE GESTURE ------------------
                gesture_buffer.append(tuple(fingers))
                if len(gesture_buffer) > BUFFER_SIZE:
                    gesture_buffer.pop(0)

                stable = max(set(gesture_buffer), key=gesture_buffer.count)

                # ------------------ COLOR SELECTION ------------------
                if stable[1] and stable[2]:
                    xp, yp = 0, 0
                    if y1 < 100:
                        index = x1 // 320
                        if index < len(colors):
                            drawColor = colors[index]

                # ------------------ DRAW MODE ------------------
                elif stable[1] and not any(stable[2:]):
                    cv2.circle(frame, (curr_x, curr_y), thickness//2, drawColor, cv2.FILLED)

                    if xp == 0 and yp == 0:
                        xp, yp = curr_x, curr_y

                    cv2.line(imgCanvas, (xp, yp), (curr_x, curr_y), drawColor, thickness)
                    xp, yp = curr_x, curr_y

                # ------------------ CLEAR ------------------
                elif all(f == 0 for f in stable):
                    imgCanvas = np.zeros((height, width, 3), np.uint8)

                # ------------------ THICKNESS CONTROL ------------------
                if stable[0] and stable[1]:
                    raw_dist = math.hypot(x1 - x3, y1 - y3)
                    target = int(np.interp(raw_dist, [20, 200], [5, 50]))
                    thickness = int(thickness * 0.7 + target * 0.3)

                    mid_x, mid_y = (x1 + x3)//2, (y1 + y3)//2
                    cv2.circle(frame, (mid_x, mid_y), thickness//2, drawColor, -1)

                # ------------------ HAND SKELETON ------------------
                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),
                    (9,13),(13,14),(14,15),(15,16),
                    (13,17),(17,18),(18,19),(19,20),
                    (0,17)
                ]

                for c in connections:
                    cv2.line(frame, tuple(points[c[0]]), tuple(points[c[1]]), (255,255,255), 2)

                for x, y in points:
                    cv2.circle(frame, (x, y), 4, (0,255,255), -1)

    # ------------------ MERGE CANVAS ------------------
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(frame, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    cv2.imshow("Virtual Painter Pro", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()