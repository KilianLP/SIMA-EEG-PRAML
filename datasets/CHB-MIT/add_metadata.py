import argparse
import os
import pickle


CHANNELS = [
    "F7-T7",
    "T7-P7",
    "F8-T8",
    "T8-P8",
]


def process_metadata(summary, filename):
    with open(summary, "r") as f:
        lines = f.readlines()

    metadata = {}
    times = []

    for i in range(len(lines)):
        line = lines[i].split()
        if len(line) == 3 and line[2] == filename:
            j = i + 1
            processed = False
            while not processed:
                if lines[j].split()[0] == "Number":
                    seizures = int(lines[j].split()[-1])
                    processed = True
                j = j + 1

            if seizures > 0:
                j = i + 1
                for _ in range(seizures):
                    processed = False
                    while not processed:
                        l = lines[j].split()
                        if l and l[0] == "Seizure" and "Start" in l:
                            start = int(l[-2]) * 256 - 1
                            end = int(lines[j + 1].split()[-2]) * 256 - 1
                            processed = True
                        j = j + 1
                    times.append((start, end))

            metadata["seizures"] = seizures
            metadata["times"] = times
            return metadata

    return {"seizures": 0, "times": []}


def parse_patient_and_edf(filename):
    """
    Expected filename format: chb02_21_4ch.pkl
    Returns: patient="chb02", edf_filename="chb02_21.edf"
    """
    base = os.path.basename(filename)
    name = base.replace(".pkl", "")
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {filename}")

    patient = parts[0]
    edf_num = parts[1]
    edf_filename = f"{patient}_{edf_num}.edf"
    return patient, edf_filename


def add_metadata_to_pkl(raw_root, pkl_root):
    for file in os.listdir(pkl_root):
        if not file.endswith(".pkl"):
            continue

        pkl_path = os.path.join(pkl_root, file)
        patient, edf_filename = parse_patient_and_edf(file)
        summary_path = os.path.join(raw_root, patient, f"{patient}-summary.txt")

        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Summary not found: {summary_path}")

        metadata = process_metadata(summary_path, edf_filename)
        metadata["channels"] = CHANNELS

        record = pickle.load(open(pkl_path, "rb"))
        record["metadata"] = metadata

        pickle.dump(record, open(pkl_path, "wb"))


def main():
    parser = argparse.ArgumentParser(
        description="Add CHB-MIT metadata into existing .pkl files."
    )
    parser.add_argument(
        "--raw",
        required=True,
        help="Path to raw CHB-MIT data root (contains chbXX folders and summaries).",
    )
    parser.add_argument(
        "--pkl",
        required=True,
        help="Path to folder containing .pkl files like chb02_21_4ch.pkl.",
    )

    args = parser.parse_args()
    add_metadata_to_pkl(args.raw, args.pkl)


if __name__ == "__main__":
    main()
