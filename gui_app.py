# ORION-X - main window (Step 8)
#
# Run:  python gui_app.py
#       python gui_app.py --profile subway_surfers
#
# F8 = control ON/OFF and F9 = quit work even when the game is in front.

import os
import sys
import glob
import json
import argparse

from core.engine import Engine          # (imported first: loads TensorFlow)
from core.dashboard import DashboardTab

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                               QPushButton, QComboBox, QProgressBar,
                               QPlainTextEdit, QHBoxLayout, QVBoxLayout,
                               QGroupBox, QTabWidget)

VOSK_MODEL = "models/vosk-model-small-en-us-0.15"

STYLE = """
QMainWindow, QWidget { background-color: #14161c; color: #e6e6e6; font-size: 14px; }
QGroupBox { border: 1px solid #2c303a; border-radius: 6px; margin-top: 10px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #8ab4ff; }
QComboBox { background: #1f222b; border: 1px solid #2c303a; padding: 6px; }
QPlainTextEdit { background: #0e1014; border: 1px solid #2c303a; font-family: Consolas; font-size: 12px; }
QProgressBar { border: 1px solid #2c303a; border-radius: 4px; text-align: center; background: #1f222b; }
QProgressBar::chunk { background: #3aa0ff; }
"""


class MainWindow(QMainWindow):
    def __init__(self, profile_files, start_index):
        super().__init__()
        self.setWindowTitle("ORION-X")
        self.resize(1100, 640)

        # ---- left: camera ----
        self.video = QLabel("Loading model and camera...")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 480)
        self.video.setStyleSheet("background: #000; border-radius: 6px;")

        # ---- right: controls ----
        self.control_btn = QPushButton()
        self.control_btn.setCheckable(True)
        self.control_btn.setMinimumHeight(48)
        self._style_control(False)

        self.profile_box = QComboBox()
        for path in profile_files:
            name = json.load(open(path)).get("name", os.path.basename(path))
            self.profile_box.addItem(name)
        self.profile_box.setCurrentIndex(start_index)

        profile_group = QGroupBox("Game profile")
        pl = QVBoxLayout(profile_group)
        pl.addWidget(self.profile_box)

        self.gesture_label = QLabel("NONE")
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #7dffa0;")
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.raw_label = QLabel("Model thinks: -")
        self.raw_label.setAlignment(Qt.AlignCenter)
        self.raw_label.setStyleSheet("color: #9aa0ab; font-size: 12px;")

        gesture_group = QGroupBox("Gesture")
        gl = QVBoxLayout(gesture_group)
        gl.addWidget(self.gesture_label)
        gl.addWidget(self.conf_bar)
        gl.addWidget(self.raw_label)

        self.say_label = QLabel("Say: -")
        self.say_label.setWordWrap(True)
        self.voice_label = QLabel("Last heard: -")
        self.voice_label.setStyleSheet("color: #ffd166; font-weight: bold;")

        voice_group = QGroupBox("Voice")
        vl = QVBoxLayout(voice_group)
        vl.addWidget(self.say_label)
        vl.addWidget(self.voice_label)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(300)

        log_group = QGroupBox("Event log")
        ll = QVBoxLayout(log_group)
        ll.addWidget(self.log_box)

        right = QVBoxLayout()
        right.addWidget(self.control_btn)
        right.addWidget(profile_group)
        right.addWidget(gesture_group)
        right.addWidget(voice_group)
        right.addWidget(log_group, 1)

        root = QHBoxLayout()
        root.addWidget(self.video, 3)
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setMaximumWidth(380)
        root.addWidget(right_widget, 1)

        central = QWidget()
        central.setLayout(root)

        self.dashboard = DashboardTab()
        tabs = QTabWidget()
        tabs.addTab(central, "Control")
        tabs.addTab(self.dashboard, "Dashboard")
        self.setCentralWidget(tabs)

        # ---- engine ----
        self.engine = Engine(profile_files, start_index, VOSK_MODEL)
        self.engine.frame_ready.connect(self.on_frame)
        self.engine.gesture_state.connect(self.on_gesture)
        self.engine.voice_heard.connect(self.on_voice)
        self.engine.profile_loaded.connect(self.on_profile)
        self.engine.control_changed.connect(self.on_control)
        self.engine.log.connect(self.add_log)
        self.engine.error.connect(self.on_error)
        self.engine.quit_requested.connect(self.close)
        self.engine.session_started.connect(self.dashboard.set_session)

        self.control_btn.clicked.connect(lambda checked: self.engine.set_control(checked))
        self.profile_box.currentIndexChanged.connect(self.engine.request_profile)

        self.engine.start()

    # ---- helpers ----
    def _style_control(self, on):
        if on:
            self.control_btn.setText("CONTROL: ON   (F8)")
            self.control_btn.setStyleSheet(
                "background: #1e8f4e; color: white; font-size: 16px; font-weight: bold; border-radius: 6px;")
        else:
            self.control_btn.setText("CONTROL: OFF   (F8)")
            self.control_btn.setStyleSheet(
                "background: #a03030; color: white; font-size: 16px; font-weight: bold; border-radius: 6px;")

    def add_log(self, text):
        self.log_box.appendPlainText(text)

    # ---- slots (called by the engine) ----
    def on_frame(self, image):
        pix = QPixmap.fromImage(image).scaled(self.video.size(), Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation)
        self.video.setPixmap(pix)

    def on_gesture(self, name, confidence, raw):
        self.gesture_label.setText(name.upper())
        self.conf_bar.setValue(int(confidence * 100))
        # what the model guesses, even when it is not sure enough to act
        self.raw_label.setText("Model thinks: " + raw + " (" + str(int(confidence * 100)) + "%)")

    def on_voice(self, word):
        self.voice_label.setText("Last heard: " + word)
        self.add_log("voice: " + word)

    def on_profile(self, name, words):
        self.say_label.setText("Say: " + ", ".join(words) if words else "Say: (no voice commands)")
        self.add_log("profile: " + name)

    def on_control(self, on):
        self.control_btn.blockSignals(True)
        self.control_btn.setChecked(on)
        self.control_btn.blockSignals(False)
        self._style_control(on)
        self.add_log("control " + ("ON" if on else "OFF"))

    def on_error(self, message):
        self.add_log("ERROR: " + message)
        self.video.setText(message)

    def closeEvent(self, event):
        self.engine.stop()
        self.engine.wait(4000)
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="default")
    args = parser.parse_args()

    profile_files = sorted(glob.glob("profiles/*.json"))
    if not profile_files:
        raise SystemExit("No profiles found in the profiles/ folder.")
    start = 0
    for i, path in enumerate(profile_files):
        if os.path.splitext(os.path.basename(path))[0] == args.profile:
            start = i

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow(profile_files, start)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()