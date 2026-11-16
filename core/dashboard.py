# ORION-X - analytics tab (charts made with PyQtGraph)

import datetime
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                               QTableWidget, QTableWidgetItem, QGroupBox,
                               QHeaderView, QAbstractItemView)

from core.database import Database

pg.setConfigOption("background", "#14161c")
pg.setConfigOption("foreground", "#c8ccd4")

GESTURE_COLOR = "#3aa0ff"
VOICE_COLOR = "#ffd166"


def make_plot(title):
    plot = pg.PlotWidget(title=title)
    plot.setMinimumHeight(170)
    plot.showGrid(y=True, alpha=0.25)
    plot.setMouseEnabled(x=False, y=False)
    plot.hideButtons()
    return plot


def draw_bars(plot, rows, color):
    plot.clear()
    names = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    if rows:
        plot.addItem(pg.BarGraphItem(x=list(range(len(rows))), height=counts,
                                     width=0.6, brush=color))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])
    plot.setXRange(-0.7, max(len(rows), 3) - 0.3)
    plot.setYRange(0, max(counts + [1]) * 1.25)


def format_time(seconds):
    seconds = int(seconds)
    return "{}:{:02d}".format(seconds // 60, seconds % 60)


class StatBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.value = QLabel("-")
        self.value.setAlignment(Qt.AlignCenter)
        self.value.setStyleSheet("font-size: 26px; font-weight: bold; color: #7dffa0;")
        layout = QVBoxLayout(self)
        layout.addWidget(self.value)

    def set(self, text):
        self.value.setText(str(text))


class DashboardTab(QWidget):
    def __init__(self):
        super().__init__()
        self.db = Database()
        self.session_id = None

        # ---- summary numbers ----
        self.box_gestures = StatBox("Gestures")
        self.box_voice = StatBox("Voice commands")
        self.box_conf = StatBox("Avg confidence")
        self.box_time = StatBox("Session time")
        top = QHBoxLayout()
        for box in (self.box_gestures, self.box_voice, self.box_conf, self.box_time):
            top.addWidget(box)

        # ---- charts ----
        self.gesture_plot = make_plot("Gestures this session")
        self.voice_plot = make_plot("Voice commands this session")
        mid = QHBoxLayout()
        mid.addWidget(self.gesture_plot)
        mid.addWidget(self.voice_plot)

        self.rate_plot = make_plot("Activity per minute")
        self.rate_plot.addLegend(offset=(10, 10))
        self.rate_plot.setLabel("bottom", "minutes since start")

        # ---- history table ----
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Started", "Length", "Gestures", "Voice", "Profiles"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(130)

        history_box = QGroupBox("Past sessions")
        hl = QVBoxLayout(history_box)
        hl.addWidget(self.table)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(mid, 2)
        layout.addWidget(self.rate_plot, 2)
        layout.addWidget(history_box, 2)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(2000)
        self.refresh()

    def set_session(self, session_id):
        self.session_id = session_id
        self.refresh()

    def refresh(self):
        try:
            self._refresh()
        except Exception as e:
            print("dashboard:", e)

    def _refresh(self):
        sid = self.session_id
        if sid is not None:
            g_rows = self.db.counts(sid, "gesture")
            v_rows = self.db.counts(sid, "voice")
            draw_bars(self.gesture_plot, g_rows, GESTURE_COLOR)
            draw_bars(self.voice_plot, v_rows, VOICE_COLOR)

            self.box_gestures.set(sum(r[1] for r in g_rows))
            self.box_voice.set(sum(r[1] for r in v_rows))
            conf = self.db.avg_confidence(sid)
            self.box_conf.set("-" if conf is None else "{}%".format(int(conf * 100)))
            start = self.db.started(sid)
            if start:
                import time
                self.box_time.set(format_time(time.time() - start))

            # activity per minute
            per_min = self.db.per_minute(sid)
            self.rate_plot.clear()
            last = max([0] + list(per_min["gesture"]) + list(per_min["voice"]))
            xs = list(range(last + 1))
            self.rate_plot.plot(xs, [per_min["gesture"].get(m, 0) for m in xs],
                                pen=pg.mkPen(GESTURE_COLOR, width=2), symbol="o",
                                symbolBrush=GESTURE_COLOR, name="gestures")
            self.rate_plot.plot(xs, [per_min["voice"].get(m, 0) for m in xs],
                                pen=pg.mkPen(VOICE_COLOR, width=2), symbol="o",
                                symbolBrush=VOICE_COLOR, name="voice")
            step = max(1, len(xs) // 10)          # whole minutes only on the bottom axis
            self.rate_plot.getAxis("bottom").setTicks([[(m, str(m)) for m in xs if m % step == 0]])

        # past sessions
        rows = self.db.history()
        self.table.setRowCount(len(rows))
        for r, (sid_, started, seconds, gestures, voice, profiles) in enumerate(rows):
            when = datetime.datetime.fromtimestamp(started).strftime("%d %b %H:%M")
            values = [when, format_time(seconds), gestures, voice, profiles or "-"]
            for c, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)
