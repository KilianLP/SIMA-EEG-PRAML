import os
import pickle
import numpy as np
from tqdm import tqdm

# ============================================================
# PARAMETERS
# ============================================================

ROOT = "/Brain/private/Clean_CHB_MIT_4c_8p/clean_four_channels"  # flat folder of .pkl files
OUT = "/Brain/private/Clean_CHB_MIT_4c_8p/clean_segments"

SAMPLING_RATE = 256
WINDOW_SIZE = 8 * SAMPLING_RATE           # 2048
STRIDE_TRAIN = WINDOW_SIZE                # 8s
STRIDE_TEST = 2 * SAMPLING_RATE           # 2s
IGNORE_POST_MINUTES = 15

CHANNELS = ["F7-T7", "T7-P7", "F8-T8", "T8-P8"]

# ============================================================
# DATA GENERATION
# ============================================================

def extract_signal(record):
    return np.array([record[ch] for ch in CHANNELS])


def label_window_fully_contained(start_idx, seizure_times):
    """
    Fenêtre positive UNIQUEMENT si entièrement contenue dans la crise.
    """
    end_idx = start_idx + WINDOW_SIZE
    for sz_start, sz_end in seizure_times:
        if start_idx >= sz_start and end_idx <= sz_end:
            return 1
    return 0


def generate_segments(record_path, output_folder, stride):
    record = pickle.load(open(record_path, "rb"))

    signal = extract_signal(record)
    seizure_times = record["metadata"]["times"]

    n_samples = signal.shape[1]

    for i in range(0, n_samples - WINDOW_SIZE + 1, stride):
        segment = signal[:, i:i + WINDOW_SIZE]
        label = label_window_fully_contained(i, seizure_times)

        out_name = os.path.basename(record_path).replace(".pkl", f"-{i}.pkl")

        pickle.dump(
            {"X": segment, "y": label},
            open(os.path.join(output_folder, out_name), "wb")
        )


def build_dataset(train_patients, test_patients):
    files = [
        f
        for f in os.listdir(ROOT)
        if f.endswith(".pkl") and os.path.isfile(os.path.join(ROOT, f))
    ]

    for file in tqdm(files):
        # filename format: chb02_21_4ch.pkl -> patient is "chb02"
        patient = file.split("_")[0]

        if patient in test_patients:
            split = "test"
            stride = STRIDE_TEST
        elif patient in train_patients:
            split = "train"
            stride = STRIDE_TRAIN
        else:
            continue

        output_folder = os.path.join(OUT, split)
        os.makedirs(output_folder, exist_ok=True)

        record_path = os.path.join(ROOT, file)
        generate_segments(record_path, output_folder, stride)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    os.makedirs(OUT, exist_ok=True)

    train_patients = ["chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08"]
    test_patients = ["chb01"]

    build_dataset(train_patients, test_patients)
