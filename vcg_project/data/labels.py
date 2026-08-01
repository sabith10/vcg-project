"""
Export patient metadata (age, sex, diagnosis) from the original PhysioNet
PTB Diagnostic ECG Database WFDB headers into a flat labels.csv.

The Kaggle CSV signal mirror used by `ptb_loader.py` has no header
metadata (age/sex/diagnosis live in the WFDB .hea comment lines, not in
the signal CSVs) — `PTBRecord.diagnosis` is currently always None. This
is a one-time, separate export against the *original* WFDB record files
to recover those labels, without needing to route the 6GB of signal data
through WFDB at training time.

Usage:
    python3 -m vcg_project.data.labels /path/to/wfdb/ptbdb --out labels.csv

Requires the (already optional) `wfdb` dependency and a local copy of
the original PhysioNet ptbdb directory (not the flattened CSV mirror).
"""

from __future__ import annotations

import csv
from pathlib import Path

from vcg_project.data.ptb_loader import extract_patient_id


def _parse_comments(comments: list[str]) -> dict[str, str]:
    """Parse WFDB header 'key: value' comment lines into a dict."""
    fields: dict[str, str] = {}
    for line in comments:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if key and value:
            fields[key] = value
    return fields


def extract_labels(wfdb_dir: str | Path) -> list[dict[str, str]]:
    """
    Parse every .hea file under wfdb_dir for age/sex/diagnosis.

    Parameters
    ----------
    wfdb_dir : str or Path
        Directory containing the original PhysioNet PTB WFDB records
        (searched recursively, e.g. patient001/s0001_re.hea).

    Returns
    -------
    list of dict
        One row per record: patient_id, record_name, age, sex, diagnosis.
    """
    import wfdb

    wfdb_dir = Path(wfdb_dir)
    hea_paths = sorted(wfdb_dir.rglob("*.hea"))

    rows = []
    for hea_path in hea_paths:
        record_path = hea_path.with_suffix("")
        try:
            header = wfdb.rdheader(str(record_path))
        except Exception as e:
            print(f"Warning: could not read header {hea_path}: {e}")
            continue

        fields = _parse_comments(header.comments or [])
        rows.append({
            "patient_id": extract_patient_id(hea_path),
            "record_name": hea_path.stem,
            "age": fields.get("age", ""),
            "sex": fields.get("sex", ""),
            "diagnosis": fields.get("reason_for_admission", fields.get("diagnosis", "")),
        })

    return rows


def write_labels_csv(rows: list[dict[str, str]], out_path: str | Path) -> None:
    """Write extracted label rows to a flat CSV file."""
    out_path = Path(out_path)
    fieldnames = ["patient_id", "record_name", "age", "sex", "diagnosis"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wfdb_dir", help="Directory containing original PTB WFDB .hea/.dat files")
    parser.add_argument("--out", default="labels.csv", help="Output CSV path")
    args = parser.parse_args()

    rows = extract_labels(args.wfdb_dir)
    write_labels_csv(rows, args.out)
    print(f"Wrote {len(rows)} label rows to {args.out}")


if __name__ == "__main__":
    main()
