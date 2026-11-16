# ORION-X - recognition engine (runs in a background thread for the GUI)
#
# Does everything orion_x.py does (camera, gestures, voice, key presses),
# but reports to the window using Qt signals instead of drawing it itself.

import json
import time
import collections
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from pynput import keyboard
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from core.features import prepare_input, get_features
from core.controller import Controller
from core.voice import VoiceListener
from core.database import Database

SEQ_LEN = 30
STABLE_NEEDED = 2
PREDICT_EVERY = 2
SWIPE_COOLDOWN = 0.4


class Engine(QThread):
    frame_ready = Signal(QImage)           # camera picture with hand drawn
    gesture_state = Signal(str, float, str)  # accepted gesture, confidence (0-1), model's raw guess
    voice_heard = Signal(str)              # a voice command was recognized
    profile_loaded = Signal(str, list)     # profile name, voice words
    control_changed = Signal(bool)         # control ON / OFF
    log = Signal(str)                      # text for the event log
    error = Signal(str)                    # something went wrong
    session_started = Signal(int)          # id of the new analytics session
    quit_requested = Signal()              # F9 was pressed

    def __init__(self, profile_files, start_index, vosk_model):
        super().__init__()
        self.profile_files = profile_files
        self.index = start_index
        self.vosk_model = vosk_model
        self.controller = None
        self.voice = None
        self.db = None
        self.running = True
        self.pending_profile = None

    # ---- called from the window (GUI thread) ----
    def set_control(self, on):
        if self.controller and self.controller.enabled != on:
            self.controller.toggle()
        self.control_changed.emit(bool(self.controller and self.controller.enabled))

    def request_profile(self, index):
        self.pending_profile = index

    def stop(self):
        self.running = False

    # ---- internal ----
    def _record(self, kind, name, confidence=None):
        """Save an event in the database (a database problem never stops the game)."""
        try:
            if self.db and self.controller:
                self.db.log(kind, name, confidence, self.controller.name, self.controller.enabled)
        except Exception as ex:
            self.log.emit("database error: " + str(ex))

    def _on_voice(self, word):
        self._record("voice", word)
        self.voice_heard.emit(word)
        if self.controller:
            self.controller.do_voice(word)

    def _on_key(self, key):
        if key == keyboard.Key.f8 and self.controller:
            self.set_control(not self.controller.enabled)
        elif key == keyboard.Key.f9:
            self.quit_requested.emit()

    def _announce_profile(self):
        words = list(self.controller.voice_actions.keys())
        self.profile_loaded.emit(self.controller.name, words)

    def run(self):
        # ---- load everything ----
        try:
            model = load_model("models/gesture_model.keras")
            labels = json.load(open("models/labels.json"))
            self.controller = Controller(self.profile_files[self.index])
        except Exception as e:
            self.error.emit("Could not load model/profile: " + str(e))
            return

        try:
            self.voice = VoiceListener(self.vosk_model,
                                       list(self.controller.voice_actions.keys()),
                                       self._on_voice)
            self.voice.start()
        except Exception as e:
            self.voice = None
            self.log.emit("Voice is OFF (" + str(e) + ")")

        try:
            self.db = Database()
            self.session_started.emit(self.db.start_session())
        except Exception as ex:
            self.db = None
            self.log.emit("Analytics is OFF (" + str(ex) + ")")

        self._announce_profile()
        self.control_changed.emit(False)
        listener = keyboard.Listener(on_press=self._on_key)
        listener.start()

        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands = mp_hands.Hands(max_num_hands=1,
                               min_detection_confidence=0.7,
                               min_tracking_confidence=0.6)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error.emit("Could not open the webcam.")
            self.running = False

        buffer = collections.deque(maxlen=SEQ_LEN)
        recent = collections.deque(maxlen=STABLE_NEEDED)
        active = "none"
        confidence = 0.0
        frame_count = 0
        last_swipe = 0.0

        while self.running:
            ok, frame = cap.read()
            if not ok:
                break

            # ---- profile switch requested by the dropdown ----
            if self.pending_profile is not None:
                new_index = self.pending_profile
                self.pending_profile = None
                try:
                    self.controller.load_profile(self.profile_files[new_index])
                    self.index = new_index
                    if self.voice:
                        self.voice.set_commands(list(self.controller.voice_actions.keys()))
                    active = "none"
                    self._announce_profile()
                except Exception as e:
                    self.error.emit("Could not load profile: " + str(e))

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(rgb, hand, mp_hands.HAND_CONNECTIONS)

            buffer.append(get_features(result))
            frame_count += 1

            if len(buffer) == SEQ_LEN and frame_count % PREDICT_EVERY == 0:
                x = prepare_input(np.array(buffer), model.input_shape[-1])[np.newaxis]
                probs = model(x, training=False).numpy()[0]
                best = int(np.argmax(probs))
                confidence = float(probs[best])

                if confidence >= self.controller.confidence:
                    recent.append(labels[best])
                else:
                    recent.append("none")

                if len(recent) == STABLE_NEEDED and len(set(recent)) == 1:
                    new_gesture = recent[0]
                    if new_gesture != active:
                        active = new_gesture

                        if active == "none":
                            self.controller.release_all()
                        elif active.startswith("swipe"):
                            if time.time() - last_swipe >= SWIPE_COOLDOWN:
                                last_swipe = time.time()
                                self.log.emit("gesture: " + active)
                                self._record("gesture", active, confidence)
                                self.controller.trigger(active)
                            buffer.clear()
                            recent.clear()
                            active = "none"
                        else:
                            self.log.emit("gesture: " + active)
                            self._record("gesture", active, confidence)
                            self.controller.trigger(active)

                self.gesture_state.emit(active, confidence, labels[best])

            h, w, _ = rgb.shape
            image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(image)

        # ---- clean up ----
        self.controller.release_all()          # never leave a key stuck down
        if self.db:
            self.db.end_session()
        if self.voice:
            self.voice.stop()
        listener.stop()
        cap.release()

