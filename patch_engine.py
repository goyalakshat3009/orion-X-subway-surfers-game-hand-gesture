# ORION-X - adds analytics (database logging) to core/engine.py
# Run once:  python patch_engine.py

import re

path = "core/engine.py"
e = open(path, encoding="utf-8").read()

if "session_started" in e:
    raise SystemExit("engine.py is already updated. Nothing to do.")


def swap(old, new):
    global e
    if old not in e:
        raise SystemExit("Could not find this text in engine.py:\n" + old)
    e = e.replace(old, new, 1)


swap("from core.voice import VoiceListener\n",
     "from core.voice import VoiceListener\nfrom core.database import Database\n")

swap("    quit_requested = Signal()",
     "    session_started = Signal(int)          # id of the new analytics session\n"
     "    quit_requested = Signal()")

swap("        self.voice = None\n        self.running = True\n",
     "        self.voice = None\n        self.db = None\n        self.running = True\n")

swap("    def _on_voice(self, word):\n        self.voice_heard.emit(word)\n",
     '''    def _record(self, kind, name, confidence=None):
        """Save an event in the database (a database problem never stops the game)."""
        try:
            if self.db and self.controller:
                self.db.log(kind, name, confidence, self.controller.name, self.controller.enabled)
        except Exception as ex:
            self.log.emit("database error: " + str(ex))

    def _on_voice(self, word):
        self._record("voice", word)
        self.voice_heard.emit(word)
''')

swap("        self._announce_profile()\n        self.control_changed.emit(False)\n",
     '''        try:
            self.db = Database()
            self.session_started.emit(self.db.start_session())
        except Exception as ex:
            self.db = None
            self.log.emit("Analytics is OFF (" + str(ex) + ")")

        self._announce_profile()
        self.control_changed.emit(False)
''')

pattern = r'^( +)self\.log\.emit\("gesture: " \+ active\)\n'
count = len(re.findall(pattern, e, flags=re.M))
e = re.sub(pattern,
           lambda m: m.group(0) + m.group(1) + 'self._record("gesture", active, confidence)\n',
           e, flags=re.M)

swap("        self.controller.release_all()          # never leave a key stuck down\n",
     "        self.controller.release_all()          # never leave a key stuck down\n"
     "        if self.db:\n            self.db.end_session()\n")

open(path, "w", encoding="utf-8").write(e)
print("Done. Gesture logging added in", count, "places.")