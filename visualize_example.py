"""
Generate two 3D VCG animations from one example beat: Ground Truth vs LSTM,
and Ground Truth vs Transformer — using the same beat for both so the two
models are directly comparable.

Loads the checkpoints + normalization stats saved by train.py, picks one
beat from a held-out (validation) patient, runs both models on it, and
de-normalizes predictions back to physical (mV) scale before plotting —
since the point of the animation is an accurate representation of the
actual 3D cardiac electric vector, not a z-scored abstraction of it.

Usage:
    python3 visualize_example.py [--checkpoint-dir ./checkpoints] [--out-dir ./viz]
"""

import argparse
import os

import numpy as np
import torch

from vcg_project.data.preprocessing import (
    preprocess_dataset_from_dir,
    split_by_patient,
    apply_normalization_stats,
)
from vcg_project.models.lstm_model import ECGToVCG_LSTM
from vcg_project.models.transformer_model import ECGToVCG_Transformer
from vcg_project.visualization.vcg_3d import create_vcg_3d_animated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/ptb/PTB diagnostic ecg database csv files")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--out-dir", default="./viz")
    parser.add_argument("--max-records", type=int, default=60,
                         help="Records to scan to find an example beat (doesn't need to be the full 549).")
    parser.add_argument("--max-beats-per-record", type=int, default=15)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beat-index", type=int, default=0,
                         help="Which beat within the validation set to visualize.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load saved normalization stats (fit on the actual training run) ──
    stats_path = os.path.join(args.checkpoint_dir, "normalization_stats.npz")
    stats = np.load(stats_path)
    ecg_stats = {"mean": stats["ecg_mean"], "std": stats["ecg_std"]}
    vcg_stats = {"mean": stats["vcg_mean"], "std": stats["vcg_std"]}

    # ── Find an example beat from a held-out patient ─────────────────────
    # Same seed as training → same patient-level train/val assignment for
    # any patient present in both runs, so this beat is genuinely from a
    # patient the models didn't train on (assuming they're in this subset).
    data = preprocess_dataset_from_dir(
        args.data_dir,
        target_len=None,
        max_beats_per_record=args.max_beats_per_record,
        max_records=args.max_records,
        seed=args.seed,
    )
    _, val_idx = split_by_patient(data["patient_ids"], val_fraction=args.val_split, seed=args.seed)

    beat_i = val_idx[args.beat_index]
    patient = data["patient_ids"][beat_i]
    ecg_beat_raw = data["ecg"][beat_i:beat_i + 1]   # [1, 12, T], physical units
    vcg_beat_raw = data["vcg"][beat_i:beat_i + 1]   # [1, 3, T], physical units
    print(f"Example beat: patient {patient}, val index {args.beat_index}, T={ecg_beat_raw.shape[-1]}")

    # ── Normalize input the same way the model was trained on ────────────
    ecg_norm = apply_normalization_stats(ecg_beat_raw, ecg_stats)
    ecg_tensor = torch.tensor(ecg_norm, dtype=torch.float32).to(device)

    # ── Load trained models ───────────────────────────────────────────────
    lstm = ECGToVCG_LSTM().to(device)
    lstm.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "lstm_ecg_to_vcg.pt"), map_location=device))
    lstm.eval()

    tf = ECGToVCG_Transformer().to(device)
    tf.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "transformer_ecg_to_vcg.pt"), map_location=device))
    tf.eval()

    with torch.no_grad():
        vcg_lstm_norm = lstm(ecg_tensor).cpu().numpy()
        vcg_tf_norm = tf(ecg_tensor).cpu().numpy()

    # ── De-normalize predictions back to physical (mV) scale ─────────────
    vcg_lstm = vcg_lstm_norm * vcg_stats["std"] + vcg_stats["mean"]
    vcg_tf = vcg_tf_norm * vcg_stats["std"] + vcg_stats["mean"]

    gt_xyz = vcg_beat_raw[0]      # [3, T]
    lstm_xyz = vcg_lstm[0]        # [3, T]
    tf_xyz = vcg_tf[0]            # [3, T]

    # ── Generate two animations from the same beat ────────────────────────
    fig_lstm = create_vcg_3d_animated(
        gt_xyz, lstm_xyz=lstm_xyz,
        title=f"VCG Loop: Ground Truth vs LSTM (patient {patient})",
    )
    fig_tf = create_vcg_3d_animated(
        gt_xyz, transformer_xyz=tf_xyz,
        title=f"VCG Loop: Ground Truth vs Transformer (patient {patient})",
    )

    lstm_path = os.path.join(args.out_dir, "vcg_lstm_vs_ground_truth.html")
    tf_path = os.path.join(args.out_dir, "vcg_transformer_vs_ground_truth.html")
    fig_lstm.write_html(lstm_path, include_plotlyjs=True)
    fig_tf.write_html(tf_path, include_plotlyjs=True)

    print(f"Wrote {lstm_path}")
    print(f"Wrote {tf_path}")


if __name__ == "__main__":
    main()
