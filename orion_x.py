# ORION-X - gesture + voice control for games
#
# Run:   python orion_x.py                     (uses profiles/default.json)
#        python orion_x.py --profile fps       (start with profiles/fps.json)
#
# F8  = turn control ON / OFF   (starts OFF so nothing types by accident)
# F10 = switch to the next game profile
# F9  = quit                    (all work even when the game window is in front)

import os
import time
import json
import glob
import argparse
import collections
import cv2
import numpy as np
import mediapipe as mp
from pynput import keyboard
from tensorflow.keras.models import load_model

from core.features import prepare_input, get_features
from core.controller import Controller
from core.voice import VoiceListener

# ---- settings ----
VOSK_MODEL = "models/vosk-model-small-en-us-0.15"
SEQ_LEN = 30
STABLE_NEEDED = 3
SWIPE_COOLDOWN = 0.8   # seconds before another swipe is accepted
PREDICT_EVERY = 2
# ------------------

# ---- choose profile ----
parser = argparse.ArgumentParser()
parser.add_argument("--profile", default="default", help="profile name (file in profiles/)")
args = parser.parse_args()

profile_files = sorted(glob.glob("profiles/*.json"))
if not profile_files:
    raise SystemExit("No profiles found in the profiles/ folder.")

profile_index = 0
for i, path in enumerate(profile_files):
    if os.path.splitext(os.path.basename(path))[0] == args.profile:
        profile_index = i

model = load_model("models/gesture_model.keras")
labels = json.load(open("models/labels.json"))
controller = Controller(profile_files[profile_index])

print("Profiles:", [os.path.basename(p)[:-5] for p in profile_files])
print("Active profile:", controller.name)
print("F8 = control ON/OFF | F10 = next profile | F9 = quit")


def on_voice(word):
    print(">>> voice:", word)
    controller.do_voice(word)


# voice is optional: if the model is missing, gestures still work
voice = None
try:
    voice = VoiceListener(VOSK_MODEL, list(controller.voice_actions.keys()), on_voice)
    voice.start()
    print("Voice commands:", list(controller.voice_actions.keys()))
except Exception as e:
    print("Voice is OFF (", e, ")")

# ---- global hotkeys ----
quit_flag = False
switch_flag = False


def on_key(key):
    global quit_flag, switch_flag
    if key == keyboard.Key.f8:
        state = controller.toggle()
        print("CONTROL", "ON" if state else "OFF")
    elif key == keyboard.Key.f9:
        quit_flag = True
    elif key == keyboard.Key.f10:
        switch_flag = True


keyboard.Listener(on_press=on_key).start()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.6)

buffer = collections.deque(maxlen=SEQ_LEN)
recent = collections.deque(maxlen=STABLE_NEEDED)
active = "none"
confidence = 0.0
frame_count = 0
last_swipe = 0.0

cap = cv2.VideoCapture(0)

while not quit_flag:
    ok, frame = cap.read()
    if not ok:
        break

    # ---- switch profile if F10 was pressed ----
    if switch_flag:
        switch_flag = False
        profile_index = (profile_index + 1) % len(profile_files)
        try:
            controller.load_profile(profile_files[profile_index])
            if voice:
                voice.set_commands(list(controller.voice_actions.keys()))
            active = "none"
            print("Switched to:", controller.name,
                  "| voice:", list(controller.voice_actions.keys()))
        except Exception as e:
            print("Could not load", profile_files[profile_index], "->", e)

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    buffer.append(get_features(result))
    frame_count += 1

    if len(buffer) == SEQ_LEN and frame_count % PREDICT_EVERY == 0:
        x = prepare_input(np.array(buffer), model.input_shape[-1])[np.newaxis]
        probs = model(x, training=False).numpy()[0]
        best = int(np.argmax(probs))
        confidence = float(probs[best])

        if confidence >= controller.confidence:
            recent.append(labels[best])
        else:
            recent.append("none")

        if len(recent) == STABLE_NEEDED and len(set(recent)) == 1:
            new_gesture = recent[0]
            if new_gesture != active:
                active = new_gesture

                if active == "none":
                    print(">>>", active)
                    controller.release_all()      # hand relaxed -> let go of keys
                elif active.startswith("swipe"):
                    # swipes are one-time events; ignore ones that come too
                    # soon (e.g. the hand moving back after a swipe)
                    if time.time() - last_swipe >= SWIPE_COOLDOWN:
                        last_swipe = time.time()
                        print(">>>", active)
                        controller.trigger(active)
                    buffer.clear()
                    recent.clear()
                    active = "none"
                else:
                    print(">>>", active)
                    controller.trigger(active)

    # ---- draw info ----
    status = "CONTROL: ON" if controller.enabled else "CONTROL: OFF (F8)"
    color = (0, 255, 0) if controller.enabled else (0, 0, 255)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, "Profile: " + controller.name + " (F10)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    cv2.putText(frame, "Gesture: " + active + "  " + str(int(confidence * 100)) + "%",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame, "Say: " + ", ".join(controller.voice_actions.keys()),
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("ORION-X", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

controller.release_all()      # never leave a key stuck down
if voice:
    voice.stop()
cap.release()
cv2.destroyAllWindows()
