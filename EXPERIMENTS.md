# Experiment log

This tracks the process of hardening the ECG → VCG pipeline and how the
LSTM/Transformer/Kors results changed at each step. Raw training logs
backing each row are in [`logs/`](logs/).

## Starting point: tech spec review

Before any training run, the original tech spec was reviewed and the
following issues were fixed in the pipeline up front (no "before" numbers
exist for these — they were fixed before the first run in this log):

- **No train/val split strategy specified.** Fixed: split by *patient*
  (`split_by_patient` in `preprocessing.py`), not by beat — beats from the
  same patient never appear in both train and val, avoiding leakage of
  patient-specific characteristics into the validation score.
- **Z-score normalization was ambiguous / per-beat.** Fixed: stats are now
  fit once on the training set only (`compute_normalization_stats`) and
  applied to val/test, so relative amplitude between beats/patients is
  preserved instead of every beat being rescaled to unit variance
  individually.
- **Fixed 256-point resampling** conflated sample count with time and
  wasn't needed — the R-peak-centered extraction window is already
  constant length in samples. Made resampling opt-in (`target_len`), off
  by default, so beats stay at native full resolution.
- **Evaluation metrics were all point-wise** (MSE/RMSE/R²/correlation) and
  said nothing about loop *shape*. Added `loop_area_error` (shoelace-area
  on the XY/YZ/XZ projections) and `max_deviation` (worst-case pointwise
  distance). Also fixed R² to be pooled across beats/time (it was
  averaged per-beat, which was unstable and inconsistent with how
  correlation was computed).
- **Diagnosis labels were unreachable.** `PTBRecord.diagnosis` existed but
  was never populated — the CSV signal mirror has no header metadata.
  Added `vcg_project/data/labels.py`, a one-time export that parses the
  original PhysioNet WFDB `.hea` headers into a flat `labels.csv`.
- **Animation frame count** scaled with raw sample count once resampling
  was made optional. `create_vcg_3d_animated` now auto-derives
  `frame_step` from beat length to target ~60 frames regardless of input
  resolution.

## Run 1 — baseline (patient split, raw beats, first 3 beats/record)

`python3 train.py --max-records 549 --epochs 50 --batch-size 32`
— 549 records, 1647 beats, 435 train / 108 val patients.
Log: [`logs/01_baseline_3beats.txt`](logs/01_baseline_3beats.txt)

| Method | MSE | R² | Corr | Loop area err | Max dev |
|---|---|---|---|---|---|
| Kors | 1.366 | -0.003 | 0.321 | 1.466 | 2.285 |
| LSTM | 1.217 | 0.122 | 0.389 | 1.706 | 2.098 |
| Transformer | 1.363 | 0.048 | 0.302 | 1.980 | 1.949 |

Both models showed train loss dropping steadily while val loss plateaued
after epoch 1–2 — an overfitting signal given ~435 patients backing
~650K-parameter models.

## Attempt: more beats/record, bigger batch → two real bugs found

Tried `--max-beats-per-record 15 --batch-size 64` to get more training
diversity per patient (matching the original spec's "GPU handles 64-batch,
10K beats" target). This surfaced two separate bugs, neither GPU-related:

1. **System RAM OOM** (confirmed via `dmesg`, not GPU VRAM — this WSL
   instance has ~7.7GB total RAM). Root cause: the pipeline loaded all 549
   raw records into memory (float64) *before* preprocessing any of them.
   Fixed: load CSVs as float32, and stream records through one at a time
   (`preprocess_dataset_from_dir`) instead of pre-loading the whole
   dataset.
2. **GPU OOM during evaluation** (`CUDA out of memory. Tried to allocate
   23.84 GiB` on a 6GB card). Root cause: post-training evaluation ran the
   *entire* validation set through the model in one unbatched forward
   pass — fine at ~300 val beats, broke at 1635. Fixed: route eval through
   the existing (already-batched) `evaluate_model` DataLoader helper
   instead of a raw `model(val_ecg)` call.

## Run 2 — after RAM + batched-eval fixes (still first-N beat selection)

`python3 train.py --max-records 549 --epochs 50 --batch-size 64 --max-beats-per-record 15`
— 8235 beats, 6600 train / 1635 val.
Log: [`logs/02_15beats_batched_eval_fixed.txt`](logs/02_15beats_batched_eval_fixed.txt)

| Method | MSE | R² | Corr | Loop area err | Max dev |
|---|---|---|---|---|---|
| Kors | 1.452 | 0.014 | 0.338 | 1.201 | 2.362 |
| LSTM | 1.312 | 0.135 | 0.385 | 2.206 | 2.116 |
| Transformer | 1.331 | 0.128 | 0.390 | 2.354 | 2.014 |

Both models reported "best epoch 0" — suspicious, since it means
validation loss never improved after the first epoch of 50. Investigating
this surfaced a third bug: `train_model` tracked `best_val_loss` /
`best_epoch` but never actually saved or restored the weights at that
epoch — the model returned after early stopping was whatever the *last*
trained epoch produced, not the best one. Fixed by snapshotting weights on
every val-loss improvement and restoring them before returning.

## Run 3 — after checkpoint-restore fix

Same command as Run 2, with the fix applied.
Log: [`logs/03_checkpoint_restore_fixed.txt`](logs/03_checkpoint_restore_fixed.txt)

| Method | MSE | R² | Corr | Loop area err | Max dev |
|---|---|---|---|---|---|
| Kors | 1.452 | 0.014 | 0.338 | 1.201 | 2.362 |
| LSTM | 1.264 | 0.179 | 0.367 | 1.812 | 2.350 |
| Transformer | 1.235 | 0.207 | 0.380 | 1.529 | 2.203 |

Restoring the real best-epoch weights meaningfully improved both models
(LSTM R² 0.135→0.179, Transformer 0.128→0.207) — the earlier numbers were
genuinely understating both models by evaluating post-overfit weights.
"Best epoch 0" persisted for both models, though — a real (not buggy)
signal that they reach peak generalization almost immediately.

## Run 4 — randomized beat sampling within each recording

`max_beats_per_record` was previously taking the *first* N beats of each
~38s recording in sequence — biasing training toward whichever ~15–20s
happened to come first (settling artifacts, one respiratory phase) and
never sampling the rest of the recording. Changed to sample N beats
uniformly at random from the whole recording (seeded, in
`preprocess_record` / `preprocess_dataset_from_dir`).

Log: [`logs/04_randomized_beat_sampling.txt`](logs/04_randomized_beat_sampling.txt)

| Method | MSE | R² | Corr | Loop area err | Max dev |
|---|---|---|---|---|---|
| Kors | 1.354 | 0.056 | 0.448 | 0.959 | 2.849 |
| LSTM | 1.207 | 0.212 | 0.378 | 0.938 | 2.845 |
| **Transformer** | **1.161** | **0.243** | **0.450** | 1.416 | 2.388 |

Best result so far. Transformer is the clear leader on every metric except
raw MSE tie-breaks, and its best epoch moved from 0 to 1 — a small but
real sign that the extra per-patient temporal diversity is helping, not
just noise.

## Experiment: LR warmup (branch `experiment/lr-warmup`)

Hypothesis: both models hitting best val loss at epoch 0 (Run 4) suggests
the optimizer takes a full-size step from batch 1 and lands in a sharp,
poorly-generalizing minimum — slowing convergence via warmup might find a
better one. Added `warmup_epochs` to `TrainConfig` (linear ramp from 1% to
100% of `--lr` before cosine decay, default 0/off — backward compatible).

`python3 train.py --max-records 549 --epochs 50 --batch-size 64 --max-beats-per-record 15 --lr 3e-5 --warmup-epochs 5`
Log: [`logs/05_lr_warmup_experiment.txt`](logs/05_lr_warmup_experiment.txt)

| Method | MSE | R² | Corr | Best epoch |
|---|---|---|---|---|
| Kors | 1.354 | 0.056 | 0.448 | — |
| LSTM | 1.212 | 0.204 | 0.398 | 6 (was 0) |
| Transformer | 1.173 | 0.235 | 0.432 | 6 (was 0) |

**Result: mechanically worked, didn't help.** Both models now train
productively for 6 epochs instead of overfitting immediately at epoch
0-1 — the warmup is doing what it's supposed to. But final R² is
marginally *lower* than Run 4 (LSTM 0.212→0.204, Transformer
0.243→0.235), not higher. If "converges too fast → sharp minimum" were
the dominant problem, slowing convergence down should have found a
better generalizing point, not just taken a slower path to a similar or
slightly worse one. This is evidence *against* optimization dynamics
being the main bottleneck, and *for* the data-diversity/patient-count
hypothesis instead — makes data augmentation (adding realistic
measurement noise to the ECG input, keeping true VCG target unchanged)
the more promising next lever rather than further LR/schedule tuning.

The `warmup_epochs` capability itself is kept (default off, so master's
default behavior is unchanged) since it's a generically useful, cheap
option even though this particular hyperparameter choice wasn't a win.

## Experiment: Gaussian noise augmentation (branch `experiment/gaussian-noise-augmentation`)

Discussion before running this: noise augmentation is a generic denoiser,
not specifically targeted at the failure mode we actually suspect (the
model latching onto patient-specific idiosyncrasies in individual leads
instead of the shared, geometry-respecting lead→VCG relationship that
`VCGLift`'s pseudo-inverse assumes). There was also a specific concern
that noise-as-regularizer tends to shrink prediction variance toward the
mean, and R² penalizes variance mismatch — so it could depress R² without
the reconstruction actually being worse in a real sense. Tested anyway,
as a deliberately separate lever from lead dropout (which more directly
targets the memorization hypothesis, planned as a follow-up).

Implementation: `noise_std` (default 0, opt-in) adds `N(0, noise_std)` to
the ECG input during training batches only — never in validation or at
eval time. Input is already z-scored to ~unit variance per channel at
that point, so `noise_std` is directly interpretable as a fraction of a
channel's own std.

`python3 train.py --max-records 549 --epochs 50 --batch-size 64 --max-beats-per-record 15 --noise-std 0.1`
Log: [`logs/06_gaussian_noise_std0.1.txt`](logs/06_gaussian_noise_std0.1.txt)

| Method | MSE | R² | Corr | Best epoch |
|---|---|---|---|---|
| Kors | 1.354 | 0.056 | 0.448 | — |
| LSTM | 1.207 | 0.212 | 0.378 | 0 (unchanged) |
| Transformer | 1.162 | 0.243 | 0.450 | 1 (unchanged) |

**Result: null.** Essentially identical to the no-augmentation baseline on
every metric, and best-epoch didn't move either — noise_std=0.1 had no
detectable effect on training dynamics in either direction. This is
inconclusive rather than confirming "noise doesn't help": 0.1 may simply
be too weak a perturbation (10% of an already-unit-variance channel) to
matter. Next: retest at a higher noise_std to get a real read.

### Retest at noise_std=0.3

Log: [`logs/07_gaussian_noise_std0.3.txt`](logs/07_gaussian_noise_std0.3.txt)

| Method | MSE | R² | Corr | Best epoch |
|---|---|---|---|---|
| Kors | 1.354 | 0.056 | 0.448 | — |
| LSTM | 1.211 | 0.209 | 0.373 | 0 (unchanged) |
| Transformer | 1.164 | 0.242 | 0.446 | 1 (unchanged) |

**Result: still null, now conclusively.** At 3x the noise magnitude,
nothing moved — not the metrics (LSTM R² 0.212→0.209, Transformer
essentially flat at 0.242 vs 0.243), and not even best-epoch, which is
the more telling signal: if noise were meaningfully perturbing the
optimization landscape at all, you'd expect *some* change in when
training peaks, not just in the final numbers. Across a 3x range with no
effect in either direction, this is a real negative result, not an
underpowered one. Conclusion: additive Gaussian noise on the ECG input is
not the lever that improves generalization here — consistent with the
concern raised before running it (generic denoiser, not targeted at
either the suspected failure mode of per-lead memorization or the
patient-diversity ceiling). Next candidates: lead dropout (targets
memorization directly), or raising `max_beats_per_record` past 15 (every
inspected record hit that cap exactly, meaning most of each ~38s
recording — likely 2-4x more beats per patient — is currently unused).

## Current state / open questions

- Both models still overfit within 1–2 epochs. Two live hypotheses, not
  yet distinguished: learning rate (1e-4) may be too high for this data
  size, or the ~15 beats/patient are similar enough to each other (same
  electrodes, same session) that ~435 patients' worth of independent
  signal is the real ceiling regardless of beat count.
- R² ≈ 0.24 (Transformer, best so far) is still fairly weak in absolute
  terms — worth treating current numbers as "pipeline is now correctly
  measuring what it claims to measure," not "the model is good yet."
- WFDB diagnosis labels have not actually been exported yet — no local
  copy of the original PhysioNet WFDB files was found on this machine
  (only the flattened CSV mirror). `labels.py` is ready to run once a copy
  is available.
