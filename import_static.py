# ORION-X - import the Zenodo "static gestures" data into our format
# Run:  python import_static.py data.zip

import os
import sys
import zipfile
import numpy as np

ZIP_PATH = sys.argv[1] if len(sys.argv) > 1 else "data.zip"
OUT_DIR = "data_external"
SEQ_LEN = 30
PER_GESTURE = 150

MAPPING = {
    "Open": "open_palm",
    "Close": "fist",
    "Pointer": "point",
    "Ok": "none",
    "Nice": "none",
}

rng = np.random.default_rng(42)

with zipfile.ZipFile(ZIP_PATH) as z:
    def find(name):
        for n in z.namelist():
            if n.endswith(name):
                return n
        raise SystemExit("File not found in zip: " + name)

    label_text = z.read(find("static_gestures_label.csv")).decode("utf-8-sig")
    class_names = [l.strip() for l in label_text.splitlines() if l.strip()]
    with z.open(find("static_gestures_v3.csv")) as f:
        table = np.loadtxt(f, delimiter=",")

class_ids = table[:, 0].astype(int)
poses = table[:, 1:].reshape(-1, 21, 2)
print("classes in file:", class_names, "| poses:", len(poses))


def make_sequence(xy):
    pts = np.zeros((21, 3))
    pts[:, :2] = xy
    if rng.random() < 0.5:
        pts[:, 0] *= -1

    wrist = np.array([rng.uniform(0.3, 0.8), rng.uniform(0.4, 0.9)])
    frames = []
    for _ in range(SEQ_LEN):
        wrist = np.clip(wrist + rng.normal(0, 0.002, 2), 0, 1)
        shaky = pts.copy()
        shaky[:, :2] += rng.normal(0, 0.01, (21, 2))
        shaky[0] = 0
        frames.append(np.concatenate([shaky.flatten(), wrist]))
    return np.array(frames)


sources = {}
for source_name, our_name in MAPPING.items():
    sources.setdefault(our_name, []).append(source_name)

total = 0
for our_name, source_names in sources.items():
    folder = os.path.join(OUT_DIR, our_name)
    os.makedirs(folder, exist_ok=True)
    start = len([f for f in os.listdir(folder) if f.endswith(".npy")])
    per_source = PER_GESTURE // len(source_names)
    count = 0
    for source_name in source_names:
        cid = class_names.index(source_name)
        candidates = np.where(class_ids == cid)[0]
        chosen = rng.choice(candidates, size=per_source, replace=False)
        for idx in chosen:
            np.save(os.path.join(folder, str(start + count) + ".npy"), make_sequence(poses[idx]))
            count += 1
    total += count
    print("  {:<10} +{} sequences  (from {})".format(our_name, count, source_names))

print("Done:", total, "sequences saved in", OUT_DIR + "/")