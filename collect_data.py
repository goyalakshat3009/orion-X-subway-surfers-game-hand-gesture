# ORION-X  Step 2 - record gesture data for LSTM training
#
# Keys:
#   1-9  choose gesture
#   s    record one sequence (30 frames)
#   a    auto mode ON/OFF (records again and again, short pause between)
#   d    delete the last saved sample of this gesture
#   q    quit
#
# Data is saved in:  data/<gesture_name>/<number>.npy

import os
import cv2
import numpy as np
import mediapipe as mp

from core.features import hand_travel

# ---- change these to your own gestures ----
GESTURES = ["fist", "open_palm", "point", "swipe_left", "swipe_right", "none"]
SEQ_LEN = 30          # frames per sequence
MAX_EMPTY = 4         # throw away sequence if hand is lost in more frames than this
PAUSE_FRAMES = 25     # pause between recordings in auto mode

# automatic quality check
STILL_GESTURES = ["fist", "open_palm", "point"]   # hand must stay still
SWIPE_GESTURES = ["swipe_left", "swipe_right"]    # hand must really move
STILL_LIMIT = 0.20    # still gestures: discard if the hand moved more than this
SWIPE_MIN = 0.25      # swipes: discard if the hand moved less than this
# -------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.6)


def get_features(result):
    """65 numbers per frame: 63 hand shape + 2 wrist position."""
    if not result.multi_hand_landmarks:
        return np.zeros(65)

    lm = result.multi_hand_landmarks[0].landmark
    pts = np.array([[p.x, p.y, p.z] for p in lm])

    wrist = pts[0].copy()          # where the hand is (needed for swipes)
    pts = pts - pts[0]             # shape only, ignore position
    biggest = np.max(np.abs(pts))
    if biggest > 0:
        pts = pts / biggest        # ignore hand size / distance

    return np.concatenate([pts.flatten(), wrist[:2]])


def count_samples(name):
    folder = os.path.join("data", name)
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".npy")])


def save_sequence(name, sequence):
    folder = os.path.join("data", name)
    os.makedirs(folder, exist_ok=True)
    number = count_samples(name)
    np.save(os.path.join(folder, str(number) + ".npy"), np.array(sequence))


def delete_last(name):
    n = count_samples(name)
    if n > 0:
        os.remove(os.path.join("data", name, str(n - 1) + ".npy"))


cap = cv2.VideoCapture(0)
current = 0            # selected gesture
recording = False
waiting_for_hand = False
auto_mode = False
pause = 0
sequence = []
message = ""

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    hand_found = result.multi_hand_landmarks is not None

    if hand_found:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    name = GESTURES[current]

    # auto mode: start a new recording after a short pause
    if auto_mode and not recording and not waiting_for_hand:
        pause += 1
        if pause > PAUSE_FRAMES:
            waiting_for_hand = True
            pause = 0

    # wait until the hand is visible, then start recording
    if waiting_for_hand:
        if name == "none" or hand_found:
            waiting_for_hand = False
            recording = True
            sequence = []
        else:
            cv2.putText(frame, "Show your hand...", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    # collect frames
    if recording:
        sequence.append(get_features(result))
        cv2.putText(frame, "RECORDING " + str(len(sequence)) + "/" + str(SEQ_LEN),
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if len(sequence) == SEQ_LEN:
            empty = sum(1 for f in sequence if not f.any())
            travel = hand_travel(sequence)
            # the "none" gesture is allowed to have empty frames
            if name != "none" and empty > MAX_EMPTY:
                message = "Discarded: hand lost in " + str(empty) + " frames"
            elif name in STILL_GESTURES and travel > STILL_LIMIT:
                message = "Discarded: hand moved too much (" + str(round(travel, 2)) + ") - keep still"
            elif name in SWIPE_GESTURES and travel < SWIPE_MIN:
                message = "Discarded: swipe too small (" + str(round(travel, 2)) + ") - swipe bigger"
            else:
                save_sequence(name, sequence)
                message = "Saved " + name + " #" + str(count_samples(name))
            print(message)
            recording = False
            sequence = []

    # text on screen
    cv2.putText(frame, "Gesture: " + name + "  (samples: " + str(count_samples(name)) + ")",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, message, (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    mode_text = "AUTO ON" if auto_mode else "auto off"
    cv2.putText(frame, "1-9 select | s record | a auto (" + mode_text + ") | d delete last | q quit",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("ORION-X - Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and not recording and not waiting_for_hand:
        waiting_for_hand = True
    elif key == ord('a'):
        auto_mode = not auto_mode
        pause = 0
    elif key == ord('d') and not recording:
        delete_last(name)
        message = "Deleted last sample of " + name
    elif ord('1') <= key <= ord('9'):
        index = key - ord('1')
        if index < len(GESTURES) and not recording:
            current = index
            auto_mode = False

cap.release()
cv2.destroyAllWindows()