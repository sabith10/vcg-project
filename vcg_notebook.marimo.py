"""\
# ECG → VCG Converter

Convert 12-lead ECG to 3D Vectorcardiogram using LSTM and Transformer models.

This notebook implements the full pipeline:
1. Load paired ECG+VCG data from the PTB database
2. Preprocess with R-peak aligned beat segmentation
3. Train two models (LSTM + Transformer) to convert ECG → VCG
4. Compare against Kors regression baseline
5. Visualize all methods as animated 3D VCG loops

## Setup

Place the PTB database in `./data/ptb/` (download from PhysioNet).
"""

import marimo

__app = marimo.App(width="full")


# ── Cell 1: Imports ──────────────────────────────────────────────
@__app.cell
def _():
    import numpy as np
    import torch
    import plotly.graph_objects as go
    import marimo as mo

    from vcg_project.data.ptb_loader import (
        load_all_records,
        ECG_LEAD_NAMES,
        VCG_LEAD_NAMES,
    )
    from vcg_project.data.preprocessing import (
        preprocess_dataset,
        preprocess_record,
        detect_r_peaks,
    )
    from vcg_project.models.kors import (
        kors_transform_beats,
        KORS_LEAD_INDICES,
    )
    from vcg_project.models.lstm_model import ECGToVCG_LSTM
    from vcg_project.models.transformer_model import ECGToVCG_Transformer
    from vcg_project.training.trainer import (
        TrainConfig,
        train_model,
        evaluate_model,
        evaluate_kors,
        compute_metrics,
    )
    from vcg_project.visualization.vcg_3d import (
        create_vcg_3d_static,
        create_vcg_3d_animated,
        create_vcg_2d_projections,
    )

    return (
        mo, np, torch, go,
        load_all_records, ECG_LEAD_NAMES, VCG_LEAD_NAMES,
        preprocess_dataset, preprocess_record, detect_r_peaks,
        kors_transform_beats, KORS_LEAD_INDICES,
        ECGToVCG_LSTM, ECGToVCG_Transformer,
        TrainConfig, train_model, evaluate_model, evaluate_kors, compute_metrics,
        create_vcg_3d_static, create_vcg_3d_animated, create_vcg_2d_projections,
    )


# ── Cell 2: Configuration ────────────────────────────────────────
@__app.cell
def _():
    mo.md("## 1. Configuration")

    config = mo.ui.dictionary({
        "data_dir": mo.ui.text(
            label="PTB data directory",
            value="./data/ptb/PTB diagnostic ecg database csv files",
        ),
        "sampling_rate": mo.ui.slider(
            start=250, stop=1000, step=250, value=500,
            label="Sampling rate (Hz)",
        ),
        "beat_length": mo.ui.slider(
            start=128, stop=512, step=128, value=256,
            label="Beat resampling length",
        ),
        "max_records": mo.ui.slider(
            start=10, stop=549, step=10, value=50,
            label="Max records to load (for quick testing)",
        ),
    })
    config
    return (config,)


# ── Cell 3: Data Loading ─────────────────────────────────────────
@__app.cell
def _(config, mo, load_all_records):
    mo.md("## 2. Load PTB Database")

    load_btn = mo.ui.button(label="Load Data", kind="run")

    load_btn
    return (load_btn,)


@__app.cell
def _(config, load_btn, load_all_records, mo):
    if load_btn.value:
        records = load_all_records(
            config["data_dir"],
            fs=config["sampling_rate"],
            max_records=config["max_records"],
        )
        mo.md(f"Loaded **{len(records)}** records from PTB database.")
    else:
        records = []
        mo.md("Click 'Load Data' to load the PTB database.")
    return (records,)


# ── Cell 4: Explore a Record ──────────────────────────────────────
@__app.cell
def _(records, mo):
    mo.md("### Explore Individual Record")

    if not records:
        mo.md("Load data first.")
        return

    record_selector = mo.ui.slider(
        start=0, stop=len(records) - 1, step=1, value=0,
        label=f"Record index (0–{len(records)-1})",
    )
    record_selector
    return (record_selector,)


@__app.cell
def _(records, record_selector, mo, np):
    if not records:
        return

    rec = records[record_selector.value]
    ecg = rec.ecg  # [12, T]
    vcg = rec.vcg  # [3, T]
    T = ecg.shape[1]
    duration_s = T / rec.fs

    mo.md(f"""
**Record:** {rec.record_name} (Patient: {rec.patient_id})
- Sampling rate: {rec.fs} Hz
- Duration: {duration_s:.1f}s ({T} samples)
- Diagnosis: {rec.diagnosis or 'N/A'}
- Age: {rec.age or 'N/A'}, Sex: {rec.sex or 'N/A'}
- ECG shape: {ecg.shape} (12 leads)
- VCG shape: {vcg.shape} (3 leads: VX, VY, VZ)
    """)
    return (rec, ecg, vcg)


# ── Cell 5: Plot ECG Traces ──────────────────────────────────────
@__app.cell
def _(ecg, ECG_LEAD_NAMES, go, mo, np):
    mo.md("#### 12-Lead ECG")

    if ecg is None:
        return

    ecg_fig = go.Figure()
    time_axis = np.arange(ecg.shape[1]) / 500  # assume 500Hz for display

    for i, name in enumerate(ECG_LEAD_NAMES):
        ecg_fig.add_trace(go.Scatter(
            x=time_axis, y=ecg[i],
            mode="lines", name=name,
            line=dict(width=0.8),
        ))

    ecg_fig.update_layout(
        title="12-Lead ECG",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=600,
        showlegend=True,
    )
    ecg_fig
    return (ecg_fig,)


# ── Cell 6: Preprocessing ────────────────────────────────────────
@__app.cell
def _(config, ecg, mo, preprocess_record, vcg):
    mo.md("## 3. Preprocess: R-Peak Detection & Beat Segmentation")

    if ecg is None:
        mo.md("Load a record first.")
        return

    preprocess_btn = mo.ui.button(label="Preprocess Record", kind="run")
    preprocess_btn
    return (preprocess_btn,)


@__app.cell
def _(config, ecg, preprocess_btn, preprocess_record, mo, vcg):
    if preprocess_btn.value and ecg is not None:
        result = preprocess_record(
            ecg, vcg,
            fs=config["sampling_rate"],
            target_len=config["beat_length"],
        )
        n_beats = result["n_beats"]
        mo.md(f"""
**Preprocessing complete:**
- Detected **{n_beats}** beats via R-peak detection
- Each beat resampled to **{config['beat_length']}** points
- ECG beats shape: `{result['ecg_beats'].shape}`
- VCG beats shape: `{result['vcg_beats'].shape}`
        """)
    else:
        result = None
        mo.md("Click 'Preprocess Record' to run R-peak detection.")
    return (result,)


# ── Cell 7: Kors Baseline ────────────────────────────────────────
@__app.cell
def _(result, mo):
    mo.md("### Kors Regression Baseline")

    if result is None or result["n_beats"] == 0:
        mo.md("Preprocess first.")
        return

    vcg_kors = kors_transform_beats(result["ecg_beats"])
    metrics_kors = compute_metrics(vcg_kors, result["vcg_beats"])

    mo.md(f"""
**Kors Transform Metrics (averaged over {result['n_beats']} beats):**
| Metric | Value |
|--------|-------|
| MSE | {metrics_kors['mse']:.6f} |
| RMSE | {metrics_kors['rmse']:.6f} |
| MAE | {metrics_kors['mae']:.6f} |
| R² | {metrics_kors['r_squared']:.4f} |
| Correlation | {metrics_kors['correlation']:.4f} |
    """)
    return (vcg_kors, metrics_kors)


# ── Cell 8: Model Selection ──────────────────────────────────────
@__app.cell
def _(mo):
    mo.md("## 4. Train Models")

    model_choice = mo.ui.radio(
        options=["LSTM", "Transformer", "Both"],
        value="Both",
        label="Model to train",
    )

    train_config = mo.ui.dictionary({
        "epochs": mo.ui.slider(start=10, stop=200, step=10, value=50, label="Epochs"),
        "lr": mo.ui.slider(start=1e-5, stop=1e-3, step=1e-5, value=1e-4, label="Learning rate"),
        "batch_size": mo.ui.slider(start=8, stop=64, step=8, value=32, label="Batch size"),
        "val_split": mo.ui.slider(start=0.1, stop=0.3, step=0.05, value=0.2, label="Validation split"),
    })

    model_choice
    train_config
    return (model_choice, train_config)


# ── Cell 9: Train ────────────────────────────────────────────────
@__app.cell
def _(model_choice, mo, train_config):
    train_btn = mo.ui.button(label="Start Training", kind="run")
    train_btn
    return (train_btn,)


@__app.cell
def _(
    result, train_btn, model_choice, train_config, mo, np, torch,
    ECGToVCG_LSTM, ECGToVCG_Transformer, TrainConfig, train_model,
):
    if not train_btn.value or result is None or result["n_beats"] == 0:
        lstm_model, transformer_model = None, None
        return

    ecg_beats = result["ecg_beats"]
    vcg_beats = result["vcg_beats"]
    N = len(ecg_beats)

    # Train/val split
    val_n = int(N * train_config["val_split"])
    perm = np.random.permutation(N)
    train_idx, val_idx = perm[val_n:], perm[:val_n]

    train_ecg, val_ecg = ecg_beats[train_idx], ecg_beats[val_idx]
    train_vcg, val_vcg = vcg_beats[train_idx], vcg_beats[val_idx]

    cfg = TrainConfig(
        lr=train_config["lr"],
        batch_size=train_config["batch_size"],
        epochs=train_config["epochs"],
    )

    lstm_model, transformer_model = None, None

    if model_choice in ("LSTM", "Both"):
        mo.md("### Training LSTM...")
        lstm_model = ECGToVCG_LSTM()
        print(f"LSTM parameters: {lstm_model.count_parameters():,}")
        state_lstm = train_model(lstm_model, train_ecg, train_vcg, val_ecg, val_vcg, cfg)
        mo.md(f"LSTM done. Best val loss: {state_lstm.best_val_loss:.6f}")

    if model_choice in ("Transformer", "Both"):
        mo.md("### Training Transformer...")
        transformer_model = ECGToVCG_Transformer()
        print(f"Transformer parameters: {transformer_model.count_parameters():,}")
        state_tf = train_model(transformer_model, train_ecg, train_vcg, val_ecg, val_vcg, cfg)
        mo.md(f"Transformer done. Best val loss: {state_tf.best_val_loss:.6f}")

    return (lstm_model, transformer_model)


# ── Cell 10: Evaluate ────────────────────────────────────────────
@__app.cell
def _(
    result, lstm_model, transformer_model, mo,
    evaluate_model, compute_metrics, kors_transform_beats,
    np, torch,
):
    mo.md("## 5. Evaluation")

    if result is None or result["n_beats"] == 0:
        mo.md("No data to evaluate.")
        return

    ecg_beats = result["ecg_beats"]
    vcg_beats = result["vcg_beats"]

    # Evaluate all methods on full dataset
    vcg_kors = kors_transform_beats(ecg_beats)
    metrics = {"Kors": compute_metrics(vcg_kors, vcg_beats)}

    if lstm_model is not None:
        with torch.no_grad():
            ecg_t = torch.tensor(ecg_beats, dtype=torch.float32)
            vcg_pred_lstm = lstm_model(ecg_t).numpy()
        metrics["LSTM"] = compute_metrics(vcg_pred_lstm, vcg_beats)

    if transformer_model is not None:
        with torch.no_grad():
            ecg_t = torch.tensor(ecg_beats, dtype=torch.float32)
            vcg_pred_tf = transformer_model(ecg_t).numpy()
        metrics["Transformer"] = compute_metrics(vcg_pred_tf, vcg_beats)

    # Build results table
    rows = []
    for method, m in metrics.items():
        rows.append({
            "Method": method,
            "MSE": f"{m['mse']:.6f}",
            "RMSE": f"{m['rmse']:.6f}",
            "MAE": f"{m['mae']:.6f}",
            "R²": f"{m['r_squared']:.4f}",
            "Correlation": f"{m['correlation']:.4f}",
        })

    mo.md(f"""
### Results Summary ({result['n_beats']} beats)

| Method | MSE | RMSE | MAE | R² | Correlation |
|--------|-----|------|-----|-----|-------------|
""" + "\n".join(
        f"| {r['Method']} | {r['MSE']} | {r['RMSE']} | {r['MAE']} | {r['R²']} | {r['Correlation']} |"
        for r in rows
    ))

    return (vcg_kors, metrics)


# ── Cell 11: Single Beat Visualization ────────────────────────────
@__app.cell
def _(
    result, vcg_kors, lstm_model, transformer_model, metrics, mo,
    create_vcg_3d_static, create_vcg_3d_animated, np, torch,
):
    mo.md("## 6. 3D VCG Visualization")

    if result is None or result["n_beats"] == 0:
        mo.md("No data to visualize.")
        return

    beat_selector = mo.ui.slider(
        start=0, stop=result["n_beats"] - 1, step=1, value=0,
        label=f"Beat index (0–{result['n_beats']-1})",
    )
    beat_selector
    return (beat_selector,)


@__app.cell
def _(
    beat_selector, result, vcg_kors, lstm_model, transformer_model, mo,
    create_vcg_3d_static, create_vcg_3d_animated, create_vcg_2d_projections,
    np, torch,
):
    if result is None or result["n_beats"] == 0:
        return

    idx = beat_selector.value
    gt = result["vcg_beats"][idx].T    # [T, 3]
    kors = vcg_kors[idx].T             # [T, 3]

    lstm_out = None
    tf_out = None

    if lstm_model is not None:
        with torch.no_grad():
            ecg_t = torch.tensor(
                result["ecg_beats"][idx:idx+1], dtype=torch.float32
            )
            lstm_out = lstm_model(ecg_t).numpy()[0].T

    if transformer_model is not None:
        with torch.no_grad():
            ecg_t = torch.tensor(
                result["ecg_beats"][idx:idx+1], dtype=torch.float32
            )
            tf_out = transformer_model(ecg_t).numpy()[0].T

    view_mode = mo.ui.radio(
        options=["Static 3D", "Animated 3D", "2D Projections"],
        value="Animated 3D",
        label="Visualization mode",
    )
    view_mode
    return (gt, kors, lstm_out, tf_out, view_mode)


@__app.cell
def _(gt, kors, lstm_out, tf_out, view_mode, mo,
       create_vcg_3d_static, create_vcg_3d_animated, create_vcg_2d_projections):
    if view_mode == "Static 3D":
        fig = create_vcg_3d_static(gt, kors, lstm_out, tf_out)
    elif view_mode == "Animated 3D":
        fig = create_vcg_3d_animated(gt, kors, lstm_out, tf_out)
    else:
        fig = create_vcg_2d_projections(gt, kors, lstm_out, tf_out)

    mo.ui.plotly(fig)
    return (fig,)


# ── Cell 12: Multi-beat View ──────────────────────────────────────
@__app.cell
def _(result, vcg_kors, lstm_model, transformer_model, mo,
       create_vcg_3d_static, np, torch):
    mo.md("### Multi-Beat VCG (averaged)")

    if result is None or result["n_beats"] < 2:
        mo.md("Need at least 2 beats.")
        return

    # Average all beats for a summary view
    gt_avg = result["vcg_beats"].mean(axis=0).T      # [T, 3]
    kors_avg = vcg_kors.mean(axis=0).T

    lstm_avg = None
    tf_avg = None

    if lstm_model is not None:
        with torch.no_grad():
            ecg_t = torch.tensor(result["ecg_beats"], dtype=torch.float32)
            lstm_avg = lstm_model(ecg_t).mean(dim=0).numpy().T

    if transformer_model is not None:
        with torch.no_grad():
            ecg_t = torch.tensor(result["ecg_beats"], dtype=torch.float32)
            tf_avg = transformer_model(ecg_t).mean(dim=0).numpy().T

    fig_avg = create_vcg_3d_static(
        gt_avg, kors_avg, lstm_avg, tf_avg,
        title="Average VCG Loop Across All Beats",
    )
    mo.ui.plotly(fig_avg)
    return


# ── Cell 13: Export ──────────────────────────────────────────────
@__app.cell
def _(lstm_model, transformer_model, mo, torch):
    mo.md("## 7. Save Models")

    save_btn = mo.ui.button(label="Save trained models")
    save_btn
    return (save_btn,)


@__app.cell
def _(save_btn, lstm_model, transformer_model, mo, torch):
    if save_btn.value:
        if lstm_model is not None:
            torch.save(lstm_model.state_dict(), "lstm_ecg_to_vcg.pt")
            mo.md("Saved `lstm_ecg_to_vcg.pt`")
        if transformer_model is not None:
            torch.save(transformer_model.state_dict(), "transformer_ecg_to_vcg.pt")
            mo.md("Saved `transformer_ecg_to_vcg.pt`")
    return


if __name__ == "__main__":
    __app.run()
