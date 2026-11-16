# ORION-X  Step 6 - test the microphone and voice commands on their own
# Say your commands (jump, reload, ...) and watch the terminal.
# Press Ctrl+C to stop.

import json
import time
from core.voice import VoiceListener

MODEL_PATH = "models/vosk-model-small-en-us-0.15"
PROFILE = "profiles/default.json"

commands = list(json.load(open(PROFILE))["voice"].keys())
print("Listening for:", commands)


def heard(word):
    print(">>> VOICE:", word)


listener = VoiceListener(MODEL_PATH, commands, heard)
listener.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    listener.stop()
    print("stopped")