import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import os
import sys
import time

# === Constants ===
PIECE_SIZE = 100
SNAP_THRESHOLD = 40
SHAPE_NAMES = ["circle", "square", "rectangle", "triangle"]
TARGETS = [(25, 100), (200, 100), (375, 100), (500, 100)]

# === Initialize pygame ===
pygame.init()

# === Load Shape Pieces ===
pieces = []
for i, name in enumerate(SHAPE_NAMES):
    try:
        path = f"shape_{name}.png"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] < 4:
            raise ValueError(f"{path} must be a PNG with transparency (alpha channel).")
        img = cv2.resize(img, (PIECE_SIZE, PIECE_SIZE))
        pieces.append({
            "name": name,
            "image": img,
            "pos": [100 + i * 120, 300],
            "target": TARGETS[i],
            "placed": False,
            "dragging": False
        })
    except Exception as e:
        print(f"❌ Failed to load shape {name}: {e}")
        sys.exit()

# === MediaPipe Hands Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Webcam Setup ===
def find_working_camera(max_index=4):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

cam_index = find_working_camera()
if cam_index is None:
    print("❌ Could not access any webcam.")
    sys.exit()

cap = cv2.VideoCapture(cam_index)
print(f"✅ Webcam {cam_index} accessed successfully.")
print("🎮 Game started! Use a pinch gesture to drag and match the shapes.")

# === Helper Functions ===
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def inside_piece(x, y, piece):
    px, py = piece["pos"]
    return px < x < px + PIECE_SIZE and py < y < py + PIECE_SIZE

def overlay_image(bg, overlay, pos):
    x, y = pos
    h, w = overlay.shape[:2]
    if y + h > bg.shape[0] or x + w > bg.shape[1]:
        return
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = bg[y:y+h, x:x+w, c] * (1 - alpha) + overlay[:, :, c] * alpha

cv2.namedWindow('Webcam', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# === Timer ===
start_time = time.time()
final_time = None
game_completed = False

# === Main Game Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    pinch = False
    pinch_point = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            thumb_tip = int(lm[4].x * w), int(lm[4].y * h)
            index_tip = int(lm[8].x * w), int(lm[8].y * h)
            middle_tip = int(lm[12].x * w), int(lm[12].y * h)

            d1 = distance(thumb_tip, index_tip)
            d2 = distance(thumb_tip, middle_tip)

            if d1 < 40 and d2 < 40:
                pinch = True
                pinch_point = ((index_tip[0] + middle_tip[0]) // 2,
                               (index_tip[1] + middle_tip[1]) // 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    for piece in pieces:
        if piece["placed"]:
            continue
        if pinch and pinch_point:
            if inside_piece(pinch_point[0], pinch_point[1], piece) or piece["dragging"]:
                piece["pos"][0] = pinch_point[0] - PIECE_SIZE // 2
                piece["pos"][1] = pinch_point[1] - PIECE_SIZE // 2
                piece["dragging"] = True
        elif piece["dragging"]:
            dx = piece["pos"][0] - piece["target"][0]
            dy = piece["pos"][1] - piece["target"][1]
            if abs(dx) < SNAP_THRESHOLD and abs(dy) < SNAP_THRESHOLD:
                piece["pos"] = list(piece["target"])
                piece["placed"] = True
            piece["dragging"] = False

    # Draw Targets
    for i, target in enumerate(TARGETS):
        shape = SHAPE_NAMES[i]
        x, y = target
        if shape == "circle":
            cv2.circle(frame, (x + 50, y + 50), 45, (0, 255, 255), 3)
        elif shape == "square":
            cv2.rectangle(frame, (x, y), (x + 100, y + 100), (255, 255, 0), 3)
        elif shape == "rectangle":
            cv2.rectangle(frame, (x, y + 25), (x + 100, y + 75), (0, 255, 255), 3)
        elif shape == "triangle":
            pts = np.array([[x + 50, y], [x + 100, y + 100], [x, y + 100]], np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 255), thickness=3)

    # Overlay pieces
    for piece in pieces:
        overlay_image(frame, piece["image"], piece["pos"])

    # Check for completion
    if not game_completed and all(p["placed"] for p in pieces):
        final_time = int(time.time() - start_time)
        game_completed = True

    # Show messages
    if game_completed:
        cv2.putText(frame, f"All Shapes Matched in {final_time} sec!", (80, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame, f"Time: {final_time}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        elapsed = int(time.time() - start_time)
        cv2.putText(frame, f"Time: {elapsed}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display frame
    cv2.imshow("Gesture Shape Matcher", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
print("👋 Game exited.")

