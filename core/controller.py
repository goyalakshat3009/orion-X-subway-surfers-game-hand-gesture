# ORION-X - turns gestures and voice commands into keyboard / mouse actions
#
# Action types (set in the profile .json file):
#   tap    press and release a key once          {"type": "tap", "key": "a"}
#   hold   keep a key down while gesture lasts   {"type": "hold", "key": "w"}
#   click  mouse click                           {"type": "click", "button": "left"}
# ("hold" works for gestures only, not for voice)

import json
import pydirectinput

pydirectinput.PAUSE = 0          # no delay between inputs
pydirectinput.FAILSAFE = False   # we use F8 as our own safety switch

ALLOWED_TYPES = ["tap", "hold", "click"]


def check_actions(actions, section, allow_hold):
    """Stop early with a clear message if a profile has a mistake."""
    for name, action in actions.items():
        kind = action.get("type")
        if kind not in ALLOWED_TYPES:
            raise ValueError("{} '{}': type must be one of {}".format(section, name, ALLOWED_TYPES))
        if kind == "hold" and not allow_hold:
            raise ValueError("{} '{}': 'hold' is not allowed for voice".format(section, name))
        if kind in ("tap", "hold") and "key" not in action:
            raise ValueError("{} '{}': missing \"key\"".format(section, name))


class Controller:
    def __init__(self, profile_path):
        self.held_keys = []        # keys we are holding right now
        self.enabled = False       # starts OFF for safety
        self.load_profile(profile_path)

    def load_profile(self, profile_path):
        """Load (or switch to) a game profile."""
        with open(profile_path) as f:
            profile = json.load(f)

        actions = profile["actions"]
        voice_actions = profile.get("voice", {})
        check_actions(actions, "actions", allow_hold=True)
        check_actions(voice_actions, "voice", allow_hold=False)

        self.release_all()                     # never carry keys into another game
        self.name = profile.get("name", "Unnamed")
        self.actions = actions
        self.voice_actions = voice_actions
        self.confidence = profile.get("confidence", 0.85)

    def trigger(self, gesture):
        """Call this when a new gesture is recognized."""
        self.release_all()                       # let go of old held keys
        action = self.actions.get(gesture)
        if action is None or not self.enabled:
            return

        kind = action["type"]
        if kind == "tap":
            pydirectinput.press(action["key"])
        elif kind == "hold":
            pydirectinput.keyDown(action["key"])
            self.held_keys.append(action["key"])
        elif kind == "click":
            pydirectinput.click(button=action.get("button", "left"))

    def do_voice(self, phrase):
        """Call this when a voice command is heard.
        Does NOT release held keys, so you can walk and jump at the same time."""
        action = self.voice_actions.get(phrase)
        if action is None or not self.enabled:
            return

        kind = action["type"]
        if kind == "tap":
            pydirectinput.press(action["key"])
        elif kind == "click":
            pydirectinput.click(button=action.get("button", "left"))

    def release_all(self):
        """Release every key we are holding."""
        for key in self.held_keys:
            pydirectinput.keyUp(key)
        self.held_keys = []

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            self.release_all()
        return self.enabled
        