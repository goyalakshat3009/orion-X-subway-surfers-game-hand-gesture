# ORION-X - shared feature helpers (used by training AND live recognition)

import numpy as np


def add_motion(seq):
    """Add wrist speed (x and y) to every frame.

    seq: array (30, 65)  ->  returns (30, 67)
    Speed = how much the wrist moved since the last frame.
    This makes swipes much easier for the LSTM to learn.
    If the hand was missing in a frame, speed is 0 for that frame.
    """
    seq = np.array(seq)
    wrist = seq[:, 63:65]
    speed = np.zeros_like(wrist)
    for i in range(1, len(seq)):
        if seq[i].any() and seq[i - 1].any():
            speed[i] = wrist[i] - wrist[i - 1]
    return np.concatenate([seq, speed * 10], axis=1)   # *10 to make it bigger


def hand_travel(seq):
    """How far the wrist moved in total during the sequence."""
    total = 0.0
    for i in range(1, len(seq)):
        if seq[i].any() and seq[i - 1].any():
            total += float(np.abs(seq[i][63:65] - seq[i - 1][63:65]).sum())
    return total


def get_features(result):
    """65 numbers per frame: 63 hand shape + 2 wrist position.
    (same as in the collector, so live data matches training data)"""
    if not result.multi_hand_landmarks:
        return np.zeros(65)

    lm = result.multi_hand_landmarks[0].landmark
    pts = np.array([[p.x, p.y, p.z] for p in lm])

    wrist = pts[0].copy()
    pts = pts - pts[0]
    biggest = np.max(np.abs(pts))
    if biggest > 0:
        pts = pts / biggest

    return np.concatenate([pts.flatten(), wrist[:2]])

def drop_z(seq):
    """Remove the 21 depth (z) columns: (30, 67) -> (30, 46)."""
    keep = [i for i in range(seq.shape[1]) if not (i < 63 and i % 3 == 2)]
    return seq[:, keep]


def prepare_input(seq, n_features):
    """Turn a recorded (30, 65) sequence into what the model expects."""
    x = add_motion(seq)
    if x.shape[1] != n_features:
        x = drop_z(x)
    return x