# ORION-X  Step 3 - train the LSTM gesture model
# Reads YOUR data from data/ and outside data from data_external/
# The model is tested ONLY on your own data.
# Saves:  models/gesture_model.keras  and  models/labels.json

import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from core.features import prepare_input

# ---- settings ----
DATA_DIR = "data"
EXTERNAL_DIR = "data_external"
USE_EXTERNAL = True       # False = train on your own data only
N_FEATURES = 46           # 46 = without z, 67 = with z
MODEL_DIR = "models"
# ------------------
os.makedirs(MODEL_DIR, exist_ok=True)


def load_folder(root, labels):
    X, y = [], []
    for index, name in enumerate(labels):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith(".npy"):
                X.append(np.load(os.path.join(folder, f)))
                y.append(index)
    return np.array(X), np.array(y)


# ---- 1. load your data ----
labels = sorted([f for f in os.listdir(DATA_DIR)
                 if os.path.isdir(os.path.join(DATA_DIR, f))])
print("gestures found:", labels)

X, y = load_folder(DATA_DIR, labels)
print("your data:", {labels[i]: int((y == i).sum()) for i in range(len(labels))})

if len(X) < 20:
    print("Not enough data. Record more samples with the collector first.")
    raise SystemExit

# ---- 2. split YOUR data: 80% train / 20% test ----
np.random.seed(42)
train_idx, test_idx = [], []
for i in range(len(labels)):
    idx = np.where(y == i)[0]
    np.random.shuffle(idx)
    n_test = max(1, int(len(idx) * 0.2))
    test_idx.extend(idx[:n_test])
    train_idx.extend(idx[n_test:])

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
print("train:", len(X_train), " test:", len(X_test))

# ---- 2b. mirror swipes: mirrored swipe_left = swipe_right (and back) ----
def mirror(seq):
    m = seq.copy()
    m[:, 0:63:3] = -m[:, 0:63:3]          # flip hand shape left-right
    m[:, 63] = 1.0 - m[:, 63]             # flip wrist position
    m[~seq.any(axis=1)] = 0               # frames with no hand stay empty
    return m


if "swipe_left" in labels and "swipe_right" in labels:
    L = labels.index("swipe_left")
    R = labels.index("swipe_right")
    extra_X, extra_y = [], []
    for s, lab in zip(X_train, y_train):
        if lab == L:
            extra_X.append(mirror(s))
            extra_y.append(R)
        elif lab == R:
            extra_X.append(mirror(s))
            extra_y.append(L)
    X_train = np.concatenate([X_train, np.array(extra_X)])
    y_train = np.concatenate([y_train, np.array(extra_y)])
    print("mirrored swipes added:", len(extra_X))

# ---- 3. noise copies of your training data ----
COPIES = 4
noisy_X, noisy_y = [X_train], [y_train]
for _ in range(COPIES):
    noise = np.random.normal(0, 0.01, X_train.shape) * (X_train != 0)
    noisy_X.append(X_train + noise)
    noisy_y.append(y_train)
X_train = np.concatenate(noisy_X)
y_train = np.concatenate(noisy_y)

# ---- 4. add outside data (training only) ----
if USE_EXTERNAL and os.path.isdir(EXTERNAL_DIR):
    X_ext, y_ext = load_folder(EXTERNAL_DIR, labels)
    if len(X_ext) > 0:
        print("outside data:", {labels[i]: int((y_ext == i).sum()) for i in range(len(labels))})
        X_train = np.concatenate([X_train, X_ext])
        y_train = np.concatenate([y_train, y_ext])
else:
    print("outside data: not used")

order = np.random.permutation(len(X_train))
X_train, y_train = X_train[order], y_train[order]
print("total training sequences:", len(X_train))

# ---- 5. model input (adds wrist speed, drops z) ----
X_train = np.array([prepare_input(s, N_FEATURES) for s in X_train])
X_test = np.array([prepare_input(s, N_FEATURES) for s in X_test])
print("model input shape:", X_train.shape[1:])

# ---- 6. build the model ----
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(len(labels), activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ---- 7. train ----
stop_early = EarlyStopping(monitor="val_loss", patience=15,
                           restore_best_weights=True)
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=100,
          batch_size=16,
          callbacks=[stop_early])

# ---- 8. check accuracy (YOUR test data only) ----
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("\nTest accuracy (your data only): {:.1f}%".format(acc * 100))

pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
for i, name in enumerate(labels):
    mask = y_test == i
    if mask.sum() > 0:
        print("  {:<12} {}/{} correct".format(name, int((pred[mask] == i).sum()), int(mask.sum())))

print("\nConfusion table (rows = real, columns = predicted):")
print("{:<13}".format("") + "".join("{:>11}".format(n[:10]) for n in labels))
for i, name in enumerate(labels):
    row = [int(((y_test == i) & (pred == j)).sum()) for j in range(len(labels))]
    print("{:<13}".format(name) + "".join("{:>11}".format(v) for v in row))

# ---- 9. save ----
model.save(os.path.join(MODEL_DIR, "gesture_model.keras"))
with open(os.path.join(MODEL_DIR, "labels.json"), "w") as f:
    json.dump(labels, f)
print("\nSaved model to models/gesture_model.keras")