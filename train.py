"""
Train ECG → VCG models: LSTM vs Transformer vs Kors baseline.

Usage:
    cd /home/sabith10/vcg_project
    python3 train.py [--max-records 100] [--epochs 50] [--batch-size 32]
"""

import argparse
import time
import sys

import numpy as np
import torch

from vcg_project.data.preprocessing import (
    preprocess_dataset_from_dir,
    split_by_patient,
    compute_normalization_stats,
    apply_normalization_stats,
)
from vcg_project.models.kors import kors_transform_beats
from vcg_project.models.lstm_model import ECGToVCG_LSTM
from vcg_project.models.transformer_model import ECGToVCG_Transformer
from vcg_project.training.trainer import (
    TrainConfig,
    train_model,
    evaluate_model,
    compute_metrics,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/ptb/PTB diagnostic ecg database csv files")
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--beat-length", type=int, default=None,
        help="If set, resample each beat to this many points. Default: "
             "keep the raw fixed-window beat at native sampling rate.",
    )
    parser.add_argument(
        "--max-beats-per-record", type=int, default=3,
        help="Keep only the first N beats per record (raw resolution, "
             "fewer beats) rather than resampling every beat in the record.",
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="./checkpoints")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 60)
    print("ECG → VCG Training")
    print("=" * 60)

    # ── Load + preprocess (streamed, one record at a time) ──────────
    # Loading all records into memory first, then preprocessing, held
    # every raw record + all extracted beats simultaneously — enough to
    # OOM on an 8GB box once beats/record goes up. Streaming keeps only
    # one raw record in memory at a time.
    t0 = time.time()
    print(
        f"\nLoading + preprocessing up to {args.max_records} records from "
        f"{args.data_dir} (beat length="
        f"{'raw/native' if args.beat_length is None else args.beat_length}, "
        f"max {args.max_beats_per_record} beats/record)..."
    )
    data = preprocess_dataset_from_dir(
        args.data_dir,
        target_len=args.beat_length,
        max_beats_per_record=args.max_beats_per_record,
        max_records=args.max_records,
        seed=args.seed,
    )
    n_beats = data["ecg"].shape[0]
    print(f"Extracted {n_beats} beats in {time.time() - t0:.1f}s")
    print(f"ECG shape: {data['ecg'].shape}")
    print(f"VCG shape: {data['vcg'].shape}")

    if n_beats == 0:
        print("No beats extracted! Check data.")
        sys.exit(1)

    # ── Train/val split (by patient, not by beat) ──────────────────
    # A per-beat random split leaks patient-specific characteristics
    # across train/val; splitting by patient means val performance
    # reflects generalization to unseen patients.
    train_idx, val_idx = split_by_patient(
        data["patient_ids"], val_fraction=args.val_split, seed=args.seed,
    )

    train_ecg_raw, val_ecg_raw = data["ecg"][train_idx], data["ecg"][val_idx]
    train_vcg_raw, val_vcg_raw = data["vcg"][train_idx], data["vcg"][val_idx]
    print(f"\nTrain: {len(train_idx)} beats | Val: {len(val_idx)} beats")
    n_train_patients = len(set(np.asarray(data["patient_ids"])[train_idx]))
    n_val_patients = len(set(np.asarray(data["patient_ids"])[val_idx]))
    print(f"Train patients: {n_train_patients} | Val patients: {n_val_patients}")

    # ── Normalize (stats fit on train only, applied to both) ────────
    # Per-beat z-scoring would rescale every beat to unit variance
    # individually, destroying real amplitude differences between beats
    # and patients. Fitting stats on train only and reusing them also
    # avoids leaking val statistics into training.
    ecg_stats = compute_normalization_stats(train_ecg_raw)
    vcg_stats = compute_normalization_stats(train_vcg_raw)

    train_ecg = apply_normalization_stats(train_ecg_raw, ecg_stats)
    val_ecg = apply_normalization_stats(val_ecg_raw, ecg_stats)
    train_vcg = apply_normalization_stats(train_vcg_raw, vcg_stats)
    val_vcg = apply_normalization_stats(val_vcg_raw, vcg_stats)

    cfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    # ── Kors baseline ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("KORS REGRESSION BASELINE")
    print("─" * 60)
    vcg_kors = kors_transform_beats(val_ecg)
    kors_metrics = compute_metrics(vcg_kors, val_vcg)
    print(f"  MSE: {kors_metrics['mse']:.6f}")
    print(f"  RMSE: {kors_metrics['rmse']:.6f}")
    print(f"  MAE: {kors_metrics['mae']:.6f}")
    print(f"  R²: {kors_metrics['r_squared']:.4f}")
    print(f"  Correlation: {kors_metrics['correlation']:.4f}")
    print(f"  Loop area error: {kors_metrics['loop_area_error']:.4f}")
    print(f"  Max deviation: {kors_metrics['max_deviation']:.4f}")

    # ── Train LSTM ─────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("TRAINING LSTM")
    print("─" * 60)
    lstm = ECGToVCG_LSTM().to(device)
    print(f"Parameters: {lstm.count_parameters():,}")
    lstm_state = train_model(lstm, train_ecg, train_vcg, val_ecg, val_vcg, cfg)

    # Batched eval (DataLoader-driven) — running the whole val set through
    # the model in one unbatched forward pass OOMs the GPU once the val
    # set is more than a couple hundred beats at T=1000.
    lstm_metrics = evaluate_model(
        lstm, val_ecg, val_vcg, batch_size=args.batch_size, device=str(device),
    )
    print(f"\n  LSTM Results:")
    print(f"  MSE: {lstm_metrics['mse']:.6f}")
    print(f"  RMSE: {lstm_metrics['rmse']:.6f}")
    print(f"  MAE: {lstm_metrics['mae']:.6f}")
    print(f"  R²: {lstm_metrics['r_squared']:.4f}")
    print(f"  Correlation: {lstm_metrics['correlation']:.4f}")
    print(f"  Loop area error: {lstm_metrics['loop_area_error']:.4f}")
    print(f"  Max deviation: {lstm_metrics['max_deviation']:.4f}")

    # ── Train Transformer ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print("TRAINING TRANSFORMER")
    print("─" * 60)
    tf = ECGToVCG_Transformer().to(device)
    print(f"Parameters: {tf.count_parameters():,}")
    tf_state = train_model(tf, train_ecg, train_vcg, val_ecg, val_vcg, cfg)

    tf_metrics = evaluate_model(
        tf, val_ecg, val_vcg, batch_size=args.batch_size, device=str(device),
    )
    print(f"\n  Transformer Results:")
    print(f"  MSE: {tf_metrics['mse']:.6f}")
    print(f"  RMSE: {tf_metrics['rmse']:.6f}")
    print(f"  MAE: {tf_metrics['mae']:.6f}")
    print(f"  R²: {tf_metrics['r_squared']:.4f}")
    print(f"  Correlation: {tf_metrics['correlation']:.4f}")
    print(f"  Loop area error: {tf_metrics['loop_area_error']:.4f}")
    print(f"  Max deviation: {tf_metrics['max_deviation']:.4f}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Method':<15} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'R²':>8} {'Corr':>8} {'AreaErr':>9} {'MaxDev':>8}")
    print("-" * 60)
    for name, m in [("Kors", kors_metrics), ("LSTM", lstm_metrics), ("Transformer", tf_metrics)]:
        print(
            f"{name:<15} {m['mse']:>10.6f} {m['rmse']:>10.6f} {m['mae']:>10.6f} "
            f"{m['r_squared']:>8.4f} {m['correlation']:>8.4f} "
            f"{m['loop_area_error']:>9.4f} {m['max_deviation']:>8.4f}"
        )
    print("=" * 60)

    # ── Save models + normalization stats ────────────────────────────
    # Stats are needed to normalize any future input the same way the
    # model was trained on, and to de-normalize predictions back to
    # physical (mV) scale for downstream use (e.g. visualization).
    import os
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(lstm.state_dict(), os.path.join(args.save_dir, "lstm_ecg_to_vcg.pt"))
    torch.save(tf.state_dict(), os.path.join(args.save_dir, "transformer_ecg_to_vcg.pt"))
    np.savez(
        os.path.join(args.save_dir, "normalization_stats.npz"),
        ecg_mean=ecg_stats["mean"], ecg_std=ecg_stats["std"],
        vcg_mean=vcg_stats["mean"], vcg_std=vcg_stats["std"],
    )
    print(f"\nModels + normalization stats saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
