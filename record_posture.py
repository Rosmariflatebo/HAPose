import csv
import os
from datetime import datetime
from mpu_translator import _read_serial, _tilt_degrees

DATASET_FILE = "posture_dataset.csv"

LABELS = {
    "1": "perfect",
    "2": "good",
    "3": "ok",
    "4": "bad",
    "5": "alarming"
}

HEADER = [
    "timestamp",
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "upper_tilt",
    "lower_tilt",
    "label"
]

def ensure_header():
    """Create file with header if missing or corrupted."""
    if not os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
        return

    # Validate existing header
    with open(DATASET_FILE, "r", newline="") as f:
        first_line = f.readline().strip()

    if first_line.replace(" ", "") != ",".join(HEADER):
        # Header is wrong → recreate file
        print("⚠️  Invalid or corrupted CSV header detected. Recreating dataset file.")
        with open(DATASET_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)

def save_sample(raw, upper_tilt, lower_tilt, label):
    ensure_header()

    with open(DATASET_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            raw["upper"]["ax"], raw["upper"]["ay"], raw["upper"]["az"],
            raw["upper"]["gx"], raw["upper"]["gy"], raw["upper"]["gz"],
            upper_tilt,
            lower_tilt,
            label
        ])

def record_session():
    print("Recording posture samples.")
    print("Choose a label:")
    print("  1 = Perfect posture")
    print("  2 = Good posture")
    print("  3 = OK posture")
    print("  4 = Bad posture")
    print("  5 = Alarming posture")
    print("  Q = Quit")

    ensure_header()

    while True:
        raw = _read_serial()
        upper_tilt = _tilt_degrees(raw["upper"])
        lower_tilt = _tilt_degrees(raw["lower"])

        key = input("Label (1/2/3/4/5/Q): ").strip().lower()

        if key == "q":
            print("Recording stopped.")
            break

        if key in LABELS:
            label = LABELS[key]
            save_sample(raw, upper_tilt, lower_tilt, label)
            print(f"Saved {label.upper()} sample.")
        else:
            print("Invalid input.")

if __name__ == "__main__":
    record_session()
