# ORION-X - check recorded data for problems
#
#  1. Still gestures (fist, open_palm, point) where the hand moved a lot
#  2. Swipes that are too small (the hand barely moved)
#
# Bad samples can be moved to data_rejected/ (nothing is deleted).
# Run:  python check_data.py

import os
import shutil
import numpy as np

DATA_DIR = "data"
STILL_GESTURES = ["fist", "open_palm", "point"]
SWIPE_GESTURES = ["swipe_left", "swipe_right"]
STILL_LIMIT = 0.15     # still gestures: hand travel must be below this
SWIPE_MIN = 0.25       # swipes: hand travel must be above this


def hand_travel(seq):
    """How far the wrist moved in total during the sequence."""
    total = 0.0
    for i in range(1, len(seq)):
        if seq[i].any() and seq[i - 1].any():
            total += np.abs(seq[i][63:65] - seq[i - 1][63:65]).sum()
    return total


print("Hand travel per gesture (bigger = more movement):")
bad = []
for name in sorted(os.listdir(DATA_DIR)):
    folder = os.path.join(DATA_DIR, name)
    if not os.path.isdir(folder):
        continue
    travels = []
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".npy"):
            continue
        t = hand_travel(np.load(os.path.join(folder, f)))
        travels.append(t)
        if name in STILL_GESTURES and t > STILL_LIMIT:
            bad.append((name, f, t, "moved too much"))
        if name in SWIPE_GESTURES and t < SWIPE_MIN:
            bad.append((name, f, t, "swipe too small"))
    if travels:
        print("  {:<12} samples {:<4} avg {:.2f}   min {:.2f}   max {:.2f}".format(
            name, len(travels), np.mean(travels), np.min(travels), np.max(travels)))

print("\nProblem samples:")
if not bad:
    print("  none found")
else:
    for name in sorted(set(b[0] for b in bad)):
        n = sum(1 for b in bad if b[0] == name)
        print("  {:<12} {} bad samples ({})".format(name, n, [b[3] for b in bad if b[0] == name][0]))
    answer = input("\nMove these {} samples to data_rejected/ ? (y/n): ".format(len(bad)))
    if answer.lower() == "y":
        for name, f, t, why in bad:
            dest = os.path.join("data_rejected", name)
            os.makedirs(dest, exist_ok=True)
            shutil.move(os.path.join(DATA_DIR, name, f), os.path.join(dest, name + "_" + f))
        print("Moved. Record new samples to replace them, then train again.")