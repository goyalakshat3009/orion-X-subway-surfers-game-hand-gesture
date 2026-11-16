# ORION-X - voice-only control (no camera, no gestures)
# Run:  python voice_only.py
# F8 = control ON / OFF  (starts OFF)       F9 = quit

import argparse
import time
from pynput import keyboard

from core.controller import Controller
from core.voice import VoiceListener

# ---- settings ----
VOSK_MODEL = "models/vosk-model-small-en-us-0.15"
COOLDOWN = 0.35        # seconds to ignore after a word (stops double firing)
STABLE_CHUNKS = 1      # 1 = fastest.  2 = safer (fewer false triggers, a bit slower)
# ------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="subway_voice")
    args = parser.parse_args()

    controller = Controller("profiles/" + args.profile + ".json")
    words = list(controller.voice_actions.keys())
    state = {"quit": False}

    def on_voice(word):
        status = "sent" if controller.enabled else "ignored (control OFF)"
        print(">>> heard:", word, "-", status)
        controller.do_voice(word)

    def on_key(key):
        if key == keyboard.Key.f8:
            print("CONTROL", "ON" if controller.toggle() else "OFF")
        elif key == keyboard.Key.f9:
            state["quit"] = True

    print("Profile:", controller.name)
    print("Loading speech model...")
    voice = VoiceListener(VOSK_MODEL, words, on_voice,
                          cooldown=COOLDOWN, stable_chunks=STABLE_CHUNKS)
    voice.start()
    keyboard.Listener(on_press=on_key).start()

    print("Say:", ", ".join(words))
    print("F8 = control ON/OFF (now OFF)  |  F9 = quit")

    try:
        while not state["quit"]:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    controller.release_all()
    voice.stop()
    print("Stopped.")


if __name__ == "__main__":
    main()