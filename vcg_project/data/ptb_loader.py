"""
PTB Diagnostic ECG Database loader.

Loads paired 12-lead ECG + Frank XYZ VCG data from the PTB database.
Each record is a CSV file with 15 columns:
  12 standard leads (I, II, III, aVR, aVL, aVF, V1-V6)
  3 Frank orthogonal leads (VX, VY, VZ)

Source: https://www.kaggle.com/datasets/abhirampolisetti/ptb-diagnostic-ecg-database
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Standard 12-lead ordering in PTB database
ECG_LEADS = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]
ECG_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Frank XYZ leads
VCG_LEADS = ["vx", "vy", "vz"]
VCG_LEAD_NAMES = ["VX", "VY", "VZ"]

# All 15 leads in CSV column order
ALL_LEADS = ECG_LEADS + VCG_LEADS

# PTB native sampling rate
PTB_FS = 1000


@dataclass
class PTBRecord:
    """A single PTB record with ECG and VCG data."""
    patient_id: str
    record_name: str
    ecg: np.ndarray          # [12, T] - 12-lead ECG
    vcg: np.ndarray          # [3, T]  - Frank XYZ VCG
    fs: int                  # Sampling frequency (Hz)
    age: str | None = None
    sex: str | None = None
    diagnosis: str | None = None


def discover_records(data_dir: str | Path) -> list[Path]:
    """
    Find all PTB CSV records in the database directory.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing .csv files.

    Returns
    -------
    list of Path
        Sorted list of CSV file paths.
    """
    data_dir = Path(data_dir)
    return sorted(data_dir.glob("*.csv"))


def extract_patient_id(csv_path: Path) -> str:
    """Derive patient ID from record filename (e.g. s0001_re.csv -> patient001)."""
    stem = csv_path.stem  # s0001_re
    # Extract numeric part: s0001 -> 1
    digits = ""
    for ch in stem:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if digits:
        return f"patient{int(digits):03d}"
    return stem


def load_record(csv_path: str | Path, fs: int | None = None) -> PTBRecord:
    """
    Load a single PTB record from CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.
    fs : int, optional
        Target sampling rate. If None, uses native 1000 Hz.

    Returns
    -------
    PTBRecord
        Named container with ecg [12, T], vcg [3, T], metadata.
    """
    csv_path = Path(csv_path)

    # Load CSV: header row + float data. float32 (not float64) since this
    # gets held in memory for every record simultaneously when batch-loading
    # the full 549-record dataset, and ECG/VCG voltage precision doesn't
    # need float64.
    signal = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    # signal shape: [T, 15]

    actual_fs = PTB_FS

    # Resample if requested
    if fs is not None and fs != actual_fs:
        from scipy.signal import resample
        n_target = int(signal.shape[0] * fs / actual_fs)
        signal = resample(signal, n_target, axis=0)
        actual_fs = fs

    # PTB lead order: first 12 = ECG, last 3 = Frank XYZ
    ecg = signal[:, :12].T   # [12, T]
    vcg = signal[:, 12:15].T # [3, T]

    return PTBRecord(
        patient_id=extract_patient_id(csv_path),
        record_name=csv_path.stem,
        ecg=ecg,
        vcg=vcg,
        fs=actual_fs,
    )


def load_all_records(
    data_dir: str | Path,
    fs: int | None = None,
    max_records: int | None = None,
) -> list[PTBRecord]:
    """
    Load all (or first N) PTB records.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing PTB CSV files.
    fs : int, optional
        Target sampling rate (Hz).
    max_records : int, optional
        Maximum number of records to load.

    Returns
    -------
    list of PTBRecord
    """
    csv_paths = discover_records(data_dir)
    if max_records is not None:
        csv_paths = csv_paths[:max_records]

    records = []
    for path in csv_paths:
        try:
            rec = load_record(path, fs=fs)
            records.append(rec)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")

    return records


def get_lead_indices() -> dict[str, int]:
    """Return mapping from lead name to index in the 15-lead array."""
    return {name: i for i, name in enumerate(ALL_LEADS)}


def get_kors_leads() -> list[int]:
    """
    Return indices of the 8 leads used by the Kors transform.

    The Kors regression uses: I, II, V1, V2, V3, V4, V5, V6
    (indices 0, 1, 6, 7, 8, 9, 10, 11 in the 12-lead ECG)
    """
    return [0, 1, 6, 7, 8, 9, 10, 11]
