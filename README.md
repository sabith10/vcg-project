# ECG → VCG Converter

Converts standard 12-lead ECG into Frank XYZ Vectorcardiogram (VCG) loops,
visualized as animated 3D loops. Three approaches are compared: a fixed
Kors regression baseline, an LSTM, and a Transformer with a physically-
grounded pseudo-inverse bottleneck (LVCG-inspired).

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the process log — what changed,
why, and how results moved at each step (includes real bugs found along
the way: a system-RAM OOM from loading the full dataset before
preprocessing, a GPU OOM from unbatched validation, and a missing
best-checkpoint restore after early stopping).

## Data

PTB Diagnostic ECG Database (Kaggle CSV mirror — 549 records, 12 standard
leads + Frank XYZ, 1000Hz). Not included in this repo (~6GB); download
separately and point `--data-dir` at it.

## Usage

```bash
# Train Kors baseline + LSTM + Transformer, compare metrics
python3 train.py --max-records 549 --epochs 50 --batch-size 64 --max-beats-per-record 15

# Generate a 3D animation comparing a trained model to ground truth
python3 visualize_example.py

# Interactive notebook
marimo edit vcg_notebook.marimo.py

# One-time export of diagnosis/age/sex labels from the original PhysioNet
# WFDB headers (not the CSV mirror) into a flat labels.csv
python3 -m vcg_project.data.labels /path/to/wfdb/ptbdb --out labels.csv
```

## Layout

| Path | Purpose |
|---|---|
| `vcg_project/data/ptb_loader.py` | CSV loader, `PTBRecord`, `discover_records` |
| `vcg_project/data/preprocessing.py` | R-peak detection, beat segmentation, patient-level split, normalization |
| `vcg_project/data/labels.py` | One-time WFDB header → `labels.csv` export |
| `vcg_project/models/kors.py` | Fixed Kors/Dower regression baseline |
| `vcg_project/models/lstm_model.py` | Conv1D + BiLSTM |
| `vcg_project/models/transformer_model.py` | Pseudo-inverse lift + self-attention + GRU |
| `vcg_project/training/trainer.py` | Training loop, early stopping, metrics |
| `vcg_project/visualization/vcg_3d.py` | Static/animated 3D loop + 2D projection plots |
| `train.py` | End-to-end training script |
| `visualize_example.py` | Generate comparison animations from a trained checkpoint |
