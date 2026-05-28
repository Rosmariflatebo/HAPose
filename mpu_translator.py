"""
MPU posture translator with 5-category dataset-driven thresholds.
"""

import math
from datetime import datetime
import csv
import serial

# ── CONFIG ──────────────────────────────────────────────────────────────────
SERIAL_PORT = "COM4"
BAUD_RATE   = 115200

# ── SERIAL INIT (lazy) ───────────────────────────────────────────────────────
_ser = None

def init_serial():
    global _ser
    if _ser is None:
        _ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# ── DATASET LOADING ──────────────────────────────────────────────────────────
def load_dataset():
    rows = []
    with open("posture_dataset.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "label" in row and row["label"].strip() != "":
                rows.append(row)
    return rows

# ── SERIAL READING ───────────────────────────────────────────────────────────
def _read_serial():
    init_serial()

    while True:
        line = _ser.readline().decode("utf-8").strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) != 6:
            continue

        try:
            v = list(map(float, parts))
        except ValueError:
            continue

        return {
            "upper": {
                "ax": v[0], "ay": v[1], "az": v[2],
                "gx": v[3], "gy": v[4], "gz": v[5]
            },
            "lower": {
                "ax": 0.0, "ay": 0.0, "az": 0.0,
                "gx": 0.0, "gy": 0.0, "gz": 0.0
            }
        }

# ── TILT CALCULATION ─────────────────────────────────────────────────────────
def _tilt_degrees(sensor):
    ax, ay, az = sensor["ax"], sensor["ay"], sensor["az"]
    magnitude = math.sqrt(ax**2 + ay**2 + az**2)
    if magnitude == 0:
        return 0.0
    return round(math.degrees(math.acos(max(-1, min(1, az / magnitude)))), 1)

# ── THRESHOLD LEARNING FROM DATASET ──────────────────────────────────────────
POSTURE_CLASSES = ["alarming", "bad", "ok", "good", "perfect"]

def compute_class_means(dataset):
    means = {}

    for c in POSTURE_CLASSES:
        values = [
            (float(r["upper_tilt"]) + float(r["lower_tilt"])) / 2
            for r in dataset
            if r["label"] == c
        ]
        if values:
            means[c] = sum(values) / len(values)

    return means

def compute_thresholds_from_means(means):
    """
    Returns:
      ordered_classes: list of (label, mean_tilt)
      thresholds: dict mapping (low_class, high_class) → midpoint tilt
    """
    ordered = sorted(means.items(), key=lambda x: x[1])
    thresholds = {}

    for i in range(len(ordered) - 1):
        c1, v1 = ordered[i]
        c2, v2 = ordered[i+1]
        thresholds[(c1, c2)] = (v1 + v2) / 2

    return ordered, thresholds

# ── LOAD DATASET + PRECOMPUTE THRESHOLDS ONCE ────────────────────────────────
DATASET = load_dataset()
MEANS, THRESHOLDS = None, None

if DATASET:
    MEANS = compute_class_means(DATASET)
    ORDERED_CLASSES, THRESHOLDS = compute_thresholds_from_means(MEANS)
else:
    ORDERED_CLASSES = []
    THRESHOLDS = {}

# ── POSTURE ANALYSIS ─────────────────────────────────────────────────────────
CLASS_SCORES = {
    "alarming": 1.0,
    "bad": 2.0,
    "ok": 3.0,
    "good": 4.0,
    "perfect": 5.0
}

def _analyse(upper_tilt, lower_tilt):
    avg = (upper_tilt + lower_tilt) / 2

    if not THRESHOLDS:
        return "No dataset loaded", 3.0

    # Walk through thresholds in order
    for (low_class, high_class), boundary in THRESHOLDS.items():
        if avg < boundary:
            return high_class.capitalize() + " posture!", CLASS_SCORES[high_class]

    # If above all boundaries → highest class
    highest_class = ORDERED_CLASSES[-1][0]
    return highest_class.capitalize() + " posture!", CLASS_SCORES[highest_class]

# ── PUBLIC API ───────────────────────────────────────────────────────────────
def get_posture():
    raw = _read_serial()
    upper_tilt = _tilt_degrees(raw["upper"])
    lower_tilt = _tilt_degrees(raw["lower"])
    status, score = _analyse(upper_tilt, lower_tilt)

    return {
        "timestamp": datetime.now(),
        "upper_tilt": upper_tilt,
        "lower_tilt": lower_tilt,
        "status": status,
        "score": score,
        "raw": raw,
    }

# ── QUICK TEST ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing MPU translator with live data...\n")
    for i in range(5):
        p = get_posture()
        print(f"[{p['timestamp'].strftime('%H:%M:%S')}]  "
              f"Upper: {p['upper_tilt']:5.1f}°  "
              f"Lower: {p['lower_tilt']:5.1f}°  "
              f"→  {p['status']}  (score {p['score']})")
