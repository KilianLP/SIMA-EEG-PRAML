import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pyedflib.highlevel as hl


def load_first_eight_records(records_root: str) -> Dict[int, List[str]]:
    """
    Read the RECORDS file in `records_root` and return a mapping of the first
    eight CHB-MIT patients to their EDF file paths.

    Keys: integers 1–8 (patients chb01 … chb08).
    Values: list of paths to that patient's EDF files (joined with records_root).
    """
    records_path = os.path.join(records_root, "RECORDS")
    if not os.path.exists(records_path):
        raise FileNotFoundError(f"RECORDS not found at {records_path}")

    records: Dict[int, List[str]] = {pid: [] for pid in range(1, 9)}

    with open(records_path, "r") as f:
        for raw_line in f:
            entry = raw_line.strip()
            if not entry:
                continue
            # Expect lines like "chb01/chb01_01.edf"
            if not entry.startswith("chb") or len(entry) < 5:
                continue
            try:
                patient_id = int(entry[3:5])
            except ValueError:
                continue

            if 1 <= patient_id <= 8:
                records[patient_id].append(os.path.join(records_root, entry))

    return records


def _parse_channel_block(lines: List[str], start_idx: int) -> (Dict[str, int], int):
    """
    Consume consecutive 'Channel X: LABEL' lines starting at start_idx.
    Returns the channel map and the index of the first non-channel line.
    """
    mapping: Dict[str, int] = {}
    idx = start_idx
    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith("Channel"):
            break
        parts = line.split(":", maxsplit=1)
        if len(parts) == 2:
            try:
                channel_num = int(parts[0].split()[1])
                label = parts[1].strip()
                mapping[label] = channel_num
            except (IndexError, ValueError):
                pass
        idx += 1
    return mapping, idx


def _channel_for_electrode(channel_map: Dict[str, int], electrode: str) -> Optional[int]:
    """Return the channel index for the given electrode label, or None if missing."""
    return channel_map.get(electrode)


def map_electrodes_per_file(
    records_root: str, electrodes: List[str]
) -> Dict[str, Dict[str, Optional[int]]]:
    """
    Parse chbXX-summary.txt files (patients 1–8) to map requested electrodes to channel
    numbers for each EDF file.

    Args:
        records_root: Root directory containing patient folders and SUMMARY files.
        electrodes: List of electrode labels to look up (e.g., ["FP1-F7", ...]).

    Returns:
        Dict keyed by EDF filename (full path). Each value is a dict
        {electrode_label: channel_number or None if absent in that file's montage}.
    """
    results: Dict[str, Dict[str, Optional[int]]] = {}

    for pid in range(1, 9):
        patient_dir = os.path.join(records_root, f"chb{pid:02d}")
        summary_path = os.path.join(patient_dir, f"chb{pid:02d}-summary.txt")
        if not os.path.exists(summary_path):
            continue

        with open(summary_path, "r") as f:
            lines = f.readlines()

        current_channels: Dict[str, int] = {}
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()

            # Identify channel definition blocks
            if line.startswith("Channels in EDF Files") or line.startswith("Channels changed"):
                idx += 1
                # skip divider lines of asterisks
                while idx < len(lines) and lines[idx].strip().startswith("*"):
                    idx += 1
                current_channels, idx = _parse_channel_block(lines, idx)
                continue

            # Map electrode channels for each file section
            if line.startswith("File Name:"):
                fname = line.split(":", maxsplit=1)[1].strip()
                file_path = os.path.join(patient_dir, fname)
                electrode_map: Dict[str, Optional[int]] = {}
                for elec in electrodes:
                    electrode_map[elec] = _channel_for_electrode(current_channels, elec)
                results[file_path] = electrode_map
                idx += 1
                continue

            idx += 1

    return results


def extract_selected_channels(
    edf_path: str,
    electrode_channels: Dict[str, Optional[int]],
    target_dir: str,
) -> str:
    """
    Load an EDF file, keep only the specified electrodes, clean NaNs to zero,
    and save the four-channel dictionary as a pickle in `target_dir`.

    Args:
        edf_path: Path to the source EDF.
        electrode_channels: Mapping {electrode_label: channel_number}. Channel
            numbers are expected to be 1-based (as in summary files); values of
            None will be filled with zeros.
        target_dir: Directory to write the cleaned pickle.

    Returns:
        Path to the written pickle file.
    """
    os.makedirs(target_dir, exist_ok=True)

    labels = list(electrode_channels.keys())  # preserve requested order

    # Collect 0-based indices for channels that exist
    ch_indices = [ch - 1 for ch in electrode_channels.values() if ch is not None]

    label_to_signal: Dict[str, np.ndarray] = {}
    if ch_indices:
        signals_read, signal_headers, _ = hl.read_edf(
            edf_path, ch_nrs=ch_indices, digital=False
        )
        for sig, header in zip(signals_read, signal_headers):
            label_to_signal[header["label"]] = np.nan_to_num(sig, copy=False)

    # Determine length from first available signal
    target_len = 0
    for sig in label_to_signal.values():
        target_len = len(sig)
        break

    clean_dict: Dict[str, np.ndarray] = {}
    for label in labels:
        sig = label_to_signal.get(label)
        if sig is None:
            clean_dict[label] = np.zeros(target_len, dtype=float)
        else:
            clean_dict[label] = sig

    out_path = os.path.join(
        target_dir, os.path.basename(edf_path).replace(".edf", "_4ch.pkl")
    )
    with open(out_path, "wb") as f:
        pickle.dump(clean_dict, f)

    return out_path


def process_first_eight_patients(
    records_root: str, electrodes: List[str], target_dir: str
) -> Dict[str, str]:
    """
    Orchestrate first-8-patient processing:
      - load EDF paths from RECORDS
      - map electrode labels to channel numbers per EDF via summaries
      - extract the requested electrodes and save cleaned pickles

    Returns:
        Dict mapping EDF path -> output pickle path (only processed files).
    """
    records = load_first_eight_records(records_root)
    electrode_map_per_file = map_electrodes_per_file(records_root, electrodes)

    outputs: Dict[str, str] = {}
    for file_list in records.values():
        for edf_path in file_list:
            channel_map = electrode_map_per_file.get(edf_path)
            if channel_map is None:
                continue  # skip if no summary-derived channels
            out_path = extract_selected_channels(edf_path, channel_map, target_dir)
            outputs[edf_path] = out_path
    return outputs


if __name__ == "__main__":
    
    root = "/Brain/private/DT_Reve_raw/CHBMIT/physionet.org/files/chbmit/1.0.0/"

    # Example electrode lookup
    example_electrodes = ["F7-T7", "T7-P7", "F8-T8", "T8-P8"]
    
    # Run end-to-end extraction for the first 8 patients
    out_dir = os.path.join("/Brain/private/Clean_CHB_MIT_4c_8p", "clean_four_channels")
    outputs = process_first_eight_patients(root, example_electrodes, out_dir)
    print(f"Extracted {len(outputs)} EDF files into {out_dir}")
