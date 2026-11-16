# ORION-X - saves every gesture and voice command in a small SQLite file
# (orionx.db, created automatically next to your scripts)

import sqlite3
import threading
import time


class Database:
    def __init__(self, path="orionx.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=5)
        self.conn.execute("PRAGMA journal_mode=WAL")     # lets the window read while we write
        self.lock = threading.Lock()
        self.session_id = None
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started REAL,
                ended REAL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                t REAL,
                kind TEXT,          -- 'gesture' or 'voice'
                name TEXT,
                confidence REAL,
                profile TEXT,
                controlled INTEGER  -- 1 if control was ON (keys were really pressed)
            );
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
        """)
        self.conn.commit()

    # ---- writing ----
    def start_session(self):
        with self.lock:
            cur = self.conn.execute("INSERT INTO sessions (started) VALUES (?)", (time.time(),))
            self.conn.commit()
            self.session_id = cur.lastrowid
            return self.session_id

    def end_session(self):
        if self.session_id is None:
            return
        with self.lock:
            self.conn.execute("UPDATE sessions SET ended = ? WHERE id = ?",
                              (time.time(), self.session_id))
            self.conn.commit()

    def log(self, kind, name, confidence=None, profile=None, controlled=False):
        if self.session_id is None:
            return
        with self.lock:
            self.conn.execute(
                "INSERT INTO events (session_id, t, kind, name, confidence, profile, controlled) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self.session_id, time.time(), kind, name, confidence, profile, int(controlled)))
            self.conn.commit()

    # ---- reading (used by the dashboard) ----
    def counts(self, session_id, kind):
        """[(name, how_many), ...] biggest first."""
        with self.lock:
            return self.conn.execute(
                "SELECT name, COUNT(*) FROM events WHERE session_id = ? AND kind = ? "
                "GROUP BY name ORDER BY COUNT(*) DESC", (session_id, kind)).fetchall()

    def started(self, session_id):
        with self.lock:
            row = self.conn.execute("SELECT started FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row[0] if row else None

    def per_minute(self, session_id):
        """{'gesture': {minute: n}, 'voice': {minute: n}} (minute 0 = session start)."""
        start = self.started(session_id)
        result = {"gesture": {}, "voice": {}}
        if start is None:
            return result
        with self.lock:
            rows = self.conn.execute(
                "SELECT kind, CAST((t - ?) / 60 AS INTEGER) AS minute, COUNT(*) FROM events "
                "WHERE session_id = ? GROUP BY kind, minute", (start, session_id)).fetchall()
        for kind, minute, n in rows:
            result[kind][minute] = n
        return result

    def avg_confidence(self, session_id):
        with self.lock:
            row = self.conn.execute(
                "SELECT AVG(confidence) FROM events WHERE session_id = ? AND kind = 'gesture'",
                (session_id,)).fetchone()
        return row[0]

    def history(self, limit=15):
        """Latest sessions: (id, started, seconds, gestures, voice, profiles)."""
        with self.lock:
            rows = self.conn.execute("""
                SELECT s.id, s.started,
                       COALESCE(s.ended, (SELECT MAX(t) FROM events e WHERE e.session_id = s.id), s.started) - s.started,
                       (SELECT COUNT(*) FROM events e WHERE e.session_id = s.id AND e.kind = 'gesture'),
                       (SELECT COUNT(*) FROM events e WHERE e.session_id = s.id AND e.kind = 'voice'),
                       (SELECT GROUP_CONCAT(DISTINCT e.profile) FROM events e WHERE e.session_id = s.id)
                FROM sessions s ORDER BY s.id DESC LIMIT ?""", (limit,)).fetchall()
        return rows
