# ORION-X  Step 4 - live gesture recognition
# Uses the model from Step 3 and your webcam.
# Press q to quit.

import json
import collections
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from core.features import add_motion, get_features

# ---- settings (play with these) ----
SEQ_LEN = 30          # must be the same as when recording
CONFIDENCE = 0.85     # model must be at least this sure (0 to 1)
STABLE_NEEDED = 3     # same answer this many times in a row before we believe it
PREDICT_EVERY = 2     # predict every 2nd frame (saves CPU)
# ------------------------------------

model = load_model("models/gesture_model.keras")
labels = json.load(open("models/labels.json"))
print("Loaded gestures:", labels)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.6)


def on_gesture(name):
    """Called once when a new gesture is recognized.
    Right now it only prints. In Step 5 this will press keys / move the mouse."""
    print(">>> GESTURE:", name)


buffer = collections.deque(maxlen=SEQ_LEN)      # last 30 frames
recent = collections.deque(maxlen=STABLE_NEEDED)  # last few predictions
active = "none"
confidence = 0.0
frame_count = 0

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    buffer.append(get_features(result))
    frame_count += 1

    # predict when we have a full sequence
    if len(buffer) == SEQ_LEN and frame_count % PREDICT_EVERY == 0:
        x = add_motion(np.array(buffer))[np.newaxis]        # shape (1, 30, 67)
        probs = model(x, training=False).numpy()[0]
        best = int(np.argmax(probs))
        confidence = float(probs[best])

        if confidence >= CONFIDENCE:
            recent.append(labels[best])
        else:
            recent.append("none")          # not sure -> treat as nothing

        # accept the gesture only if the last few answers agree
        if len(recent) == STABLE_NEEDED and len(set(recent)) == 1:
            new_gesture = recent[0]
            if new_gesture != active:
                active = new_gesture
                if active != "none":
                    on_gesture(active)
                    # swipes are one-time events: clear memory so it
                    # doesn't trigger twice for the same swipe
                    if active.startswith("swipe"):
                        buffer.clear()
                        recent.clear()
                        active = "none"

    # ---- draw info ----
    if len(buffer) < SEQ_LEN:
        cv2.putText(frame, "Getting ready...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    else:
        cv2.putText(frame, "Gesture: " + active, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, "Confidence: " + str(int(confidence * 100)) + "%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imshow("ORION-X - Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()