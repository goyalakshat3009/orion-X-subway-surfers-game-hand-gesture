# ORION-X - offline voice commands using Vosk
#
# Listens to the microphone in a background thread.
# Only the words you put in the profile are recognized (this makes it fast
# and much more accurate than open dictation).

import json
import queue
import threading
import time
import sounddevice as sd
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(-1)          # hide Vosk's long log messages

# Everyday words that are allowed but ignored. With only 2-3 command words,
# Vosk tends to force any sound into one of them; these give it a better match.
DECOYS = ["the", "a", "and", "yes", "no", "okay", "hello", "what", "this",
          "that", "is", "it", "you", "go", "now", "one", "two", "three"]


class VoiceListener:
    def __init__(self, model_path, commands, on_command,
                 cooldown=1.0, stable_chunks=2):
        """
        model_path    : folder of the Vosk model
        commands      : list of words to listen for, e.g. ["jump", "reload"]
        on_command    : function called with the word that was heard
        cooldown      : seconds to ignore everything after a command fires
                        (stops the same word firing twice)
        stable_chunks : how many 0.25s audio chunks in a row must contain the
                        word before we believe it (stops noise blips).
                        1 = fastest but more false triggers, 3 = safest.
        """
        self.on_command = on_command
        self.cooldown = cooldown
        self.stable_chunks = stable_chunks

        self.model = Model(model_path)
        self.set_commands(commands)

        self.audio_queue = queue.Queue()
        self.running = False
        self.stream = None
        self.last_fire = 0.0
        self.last_heard = ""
        self.heard_count = 0

    def set_commands(self, commands):
        """Change the words we listen for (used when switching game profiles)."""
        self.commands = [c.lower() for c in commands]
        decoys = [d for d in DECOYS if d not in self.commands]
        grammar = json.dumps(self.commands + decoys + ["[unk]"])
        self.rec = KaldiRecognizer(self.model, 16000, grammar)

    def _audio_callback(self, indata, frames, time_info, status):
        self.audio_queue.put(bytes(indata))

    def start(self):
        self.running = True
        self.stream = sd.RawInputStream(samplerate=16000, blocksize=4000,
                                        dtype="int16", channels=1,
                                        callback=self._audio_callback)
        self.stream.start()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            try:
                data = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            rec = self.rec          # (can be replaced by set_commands)
            if rec.AcceptWaveform(data):
                heard = json.loads(rec.Result()).get("text", "")
                is_final = True
            else:
                heard = json.loads(rec.PartialResult()).get("partial", "")
                is_final = False

            if heard not in self.commands:
                self.last_heard = ""
                self.heard_count = 0
                continue

            # a finished sentence is trusted; a partial must repeat a few times
            if is_final:
                confirmed = True
            else:
                if heard == self.last_heard:
                    self.heard_count += 1
                else:
                    self.last_heard = heard
                    self.heard_count = 1
                confirmed = self.heard_count >= self.stable_chunks

            if not confirmed:
                continue

            now = time.time()
            if now - self.last_fire >= self.cooldown:
                self.last_fire = now
                self.on_command(heard)

            rec.Reset()              # forget the word (also during cooldown)
            self.last_heard = ""
            self.heard_count = 0

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()