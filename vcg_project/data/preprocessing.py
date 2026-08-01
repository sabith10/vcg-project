"""
ECG/VCG preprocessing pipeline.

Handles R-peak detection, beat segmentation, and resampling
for the ECG→VCG conversion task.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample


def _pan_tompkins_detect(signal_1d: np.ndarray, fs: int = 1000) -> np.ndarray:
    """
    Pan-Tompkins style R-peak detection.

    Pure scipy implementation — no external dependencies beyond numpy/scipy.
    """
    from scipy.signal import butter, filtfilt, find_peaks

    # Bandpass filter 5-15 Hz (QRS energy band)
    low = 5.0 / (fs / 2)
    high = 15.0 / (fs / 2)
    b, a = butter(2, [low, high], btype="band")
    filtered = filtfilt(b, a, signal_1d)

    # Squaring to emphasize peaks
    squared = filtered ** 2

    # Moving average integration (window ~150ms)
    win = int(0.150 * fs)
    kernel = np.ones(win) / win
    integrated = np.convolve(squared, kernel, mode="same")

    # Adaptive thresholding
    med = np.median(integrated)
    threshold = 0.5 * med

    # Minimum distance between R-peaks (~200ms = refractory period)
    min_dist = int(0.200 * fs)

    peaks, _ = find_peaks(integrated, height=threshold, distance=min_dist)

    # Refine: find actual maximum in a small window around each detected peak
    refined = []
    half_win = int(0.050 * fs)  # 50ms window
    for p in peaks:
        start = max(0, p - half_win)
        end = min(len(signal_1d), p + half_win)
        local_max = start + np.argmax(signal_1d[start:end])
        refined.append(local_max)

    return np.array(refined, dtype=int)


def detect_r_peaks(signal_1d: np.ndarray, fs: int = 1000) -> np.ndarray:
    """
    Detect R-peaks in a single ECG lead.

    Tries NeuroKit2 first, falls back to Pan-Tompkins if unavailable.

    Parameters
    ----------
    signal_1d : np.ndarray
        1D ECG signal (typically lead II).
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Array of R-peak sample indices.
    """
    try:
        import neurokit2 as nk
        cleaned = nk.ecg_clean(signal_1d, sampling_rate=fs)
        _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
        return rpeaks["ECG_R_Peaks"]
    except (ImportError, TypeError):
        return _pan_tompkins_detect(signal_1d, fs=fs)


def segment_beats(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int = 1000,
    before_ms: int = 400,
    after_ms: int = 600,
) -> list[np.ndarray]:
    """
    Segment a continuous signal into individual beats centered on R-peaks.

    Parameters
    ----------
    signal : np.ndarray
        Signal to segment, shape [n_channels, T] or [T].
    r_peaks : np.ndarray
        R-peak sample indices.
    fs : int
        Sampling frequency.
    before_ms : int
        Milliseconds before R-peak to include.
    after_ms : int
        Milliseconds after R-peak to include.

    Returns
    -------
    list of np.ndarray
        List of beat segments. Each has shape [n_channels, beat_len]
        or [beat_len] for 1D input.
    """
    before_samples = int(fs * before_ms / 1000)
    after_samples = int(fs * after_ms / 1000)
    T = signal.shape[-1]

    beats = []
    for r_idx in r_peaks:
        start = r_idx - before_samples
        end = r_idx + after_samples

        # Skip beats that extend beyond signal boundaries
        if start < 0 or end > T:
            continue

        if signal.ndim == 2:
            beat = signal[:, start:end]
        else:
            beat = signal[start:end]

        beats.append(beat)

    return beats


def resample_beat(
    beat: np.ndarray,
    target_len: int = 256,
) -> np.ndarray:
    """
    Resample a single beat to a fixed length using FFT-based interpolation.

    Uses scipy.signal.resample which performs bandlimited interpolation
    via FFT — this preserves sharp peaks (QRS complex) unlike averaging
    or decimation.

    Parameters
    ----------
    beat : np.ndarray
        Beat segment, shape [n_channels, beat_len] or [beat_len].
    target_len : int
        Desired output length.

    Returns
    -------
    np.ndarray
        Resampled beat with length target_len.
    """
    return resample(beat, target_len, axis=-1)


def resample_beats(
    beats: list[np.ndarray],
    target_len: int = 256,
) -> np.ndarray:
    """
    Resample a list of beats to fixed length and stack.

    Parameters
    ----------
    beats : list of np.ndarray
        Each beat has shape [n_channels, beat_len].
    target_len : int
        Target length for all beats.

    Returns
    -------
    np.ndarray
        Stacked resampled beats, shape [N_beats, n_channels, target_len].
    """
    resampled = [resample_beat(b, target_len) for b in beats]
    return np.stack(resampled, axis=0)


def normalize_beat(beat: np.ndarray) -> np.ndarray:
    """
    Z-score normalize a single beat across the time axis (per-beat stats).

    NOTE: this rescales every beat to unit variance individually, which
    destroys absolute amplitude differences between beats/patients (e.g.
    low-voltage vs. hypertrophy voltage patterns). Prefer
    `compute_normalization_stats` + `apply_normalization_stats` fit on the
    training set only, unless you specifically want per-beat shape-only
    normalization.

    Parameters
    ----------
    beat : np.ndarray
        Shape [n_channels, beat_len].

    Returns
    -------
    np.ndarray
        Normalized beat with zero mean and unit variance per channel.
    """
    mean = beat.mean(axis=-1, keepdims=True)
    std = beat.std(axis=-1, keepdims=True) + 1e-8
    return (beat - mean) / std


def compute_normalization_stats(beats: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute per-channel mean/std across a whole set of beats.

    Intended to be called once on the *training* split only; apply the
    returned stats to train/val/test with `apply_normalization_stats` so
    relative amplitude differences between beats/patients are preserved
    (unlike per-beat normalization, which rescales every beat to unit
    variance individually).

    Parameters
    ----------
    beats : np.ndarray
        Shape [N, n_channels, T].

    Returns
    -------
    dict with 'mean' and 'std', each shape [1, n_channels, 1].
    """
    mean = beats.mean(axis=(0, 2), keepdims=True)
    std = beats.std(axis=(0, 2), keepdims=True) + 1e-8
    return {"mean": mean, "std": std}


def apply_normalization_stats(
    beats: np.ndarray,
    stats: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Apply previously-fit per-channel mean/std to a set of beats.

    Parameters
    ----------
    beats : np.ndarray
        Shape [N, n_channels, T].
    stats : dict
        Output of `compute_normalization_stats`.

    Returns
    -------
    np.ndarray
        Normalized beats, same shape as input.
    """
    return (beats - stats["mean"]) / stats["std"]


def preprocess_record(
    ecg: np.ndarray,
    vcg: np.ndarray,
    fs: int = 1000,
    target_len: int | None = None,
    before_ms: int = 400,
    after_ms: int = 600,
    max_beats: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Full preprocessing pipeline for a single record.

    Segmentation extracts a fixed absolute-time window around every R-peak
    (before_ms + after_ms), so beats already come out at a constant sample
    count at the native sampling rate — no resampling is required for
    batching. Resampling is therefore opt-in: pass `target_len` if you
    specifically want a lower time-resolution/fixed-point-count beat (e.g.
    for a lighter-weight model or fewer animation frames downstream);
    leave it None to keep the raw, full-resolution beat.

    Each beat is already modeled fully independently — the model's forward
    pass only sees the ~1s window of a single beat, with no recurrence or
    attention across beat boundaries — so there's no "beat-to-beat" leakage
    at the architecture level. But when `max_beats` truncates a record's
    beats, which beats get kept still matters: taking the first N in
    sequence only samples the first N*(before_ms+after_ms) of the
    recording, biasing training toward whatever's happening early (settling
    artifacts, a specific respiratory phase) and missing anything later in
    the ~38s recording. Pass `rng` to sample `max_beats` beats uniformly at
    random from the whole recording instead.

    Normalization is NOT applied here — fit `compute_normalization_stats`
    on the training split only and apply it to all splits afterwards, so
    normalization stats don't leak from val/test into training and beat
    amplitude isn't destroyed per-beat.

    Parameters
    ----------
    ecg : np.ndarray
        12-lead ECG, shape [12, T].
    vcg : np.ndarray
        Frank XYZ VCG, shape [3, T].
    fs : int
        Sampling frequency.
    target_len : int, optional
        If given, resample each beat to this many points. If None
        (default), keep the raw fixed-window beat length.
    before_ms : int
        Ms before R-peak.
    after_ms : int
        Ms after R-peak.
    max_beats : int, optional
        If given, keep only N valid beats from this record instead of all
        of them (trades beat count for keeping every kept beat at full raw
        resolution).
    rng : np.random.Generator, optional
        If given (and max_beats is set), the N beats are sampled uniformly
        at random from the whole recording. If None, the first N beats are
        kept (biased toward the start of the recording).

    Returns
    -------
    dict with keys:
        'ecg_beats': [N, 12, beat_len]
        'vcg_beats': [N, 3, beat_len]
        'r_peaks': original R-peak indices
        'n_beats': number of valid beats
    """
    # Detect R-peaks on lead II (index 1)
    r_peaks = detect_r_peaks(ecg[1], fs=fs)

    # Segment beats from both ECG and VCG (simultaneous)
    ecg_beats = segment_beats(ecg, r_peaks, fs, before_ms, after_ms)
    vcg_beats = segment_beats(vcg, r_peaks, fs, before_ms, after_ms)

    # Ensure same number of beats (should be identical since same r_peaks)
    n_beats = min(len(ecg_beats), len(vcg_beats))
    ecg_beats = ecg_beats[:n_beats]
    vcg_beats = vcg_beats[:n_beats]

    if max_beats is not None and n_beats > max_beats:
        if rng is not None:
            keep = np.sort(rng.choice(n_beats, size=max_beats, replace=False))
        else:
            keep = np.arange(max_beats)
        ecg_beats = [ecg_beats[i] for i in keep]
        vcg_beats = [vcg_beats[i] for i in keep]
        n_beats = len(ecg_beats)

    if n_beats == 0:
        return {
            "ecg_beats": np.array([]),
            "vcg_beats": np.array([]),
            "r_peaks": r_peaks,
            "n_beats": 0,
        }

    # Stack: [N, n_channels, beat_len]
    ecg_stacked = np.stack(ecg_beats, axis=0)
    vcg_stacked = np.stack(vcg_beats, axis=0)

    # Resampling is opt-in (see docstring) — off by default.
    if target_len is not None:
        ecg_stacked = resample_beats(ecg_stacked, target_len)
        vcg_stacked = resample_beats(vcg_stacked, target_len)

    return {
        "ecg_beats": ecg_stacked,    # [N, 12, beat_len]
        "vcg_beats": vcg_stacked,    # [N, 3, beat_len]
        "r_peaks": r_peaks,
        "n_beats": n_beats,
    }


def preprocess_dataset(
    records: list,
    fs: int = 1000,
    target_len: int | None = None,
    max_beats_per_record: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Preprocess all records and concatenate beats.

    Normalization is deliberately not applied here — see
    `preprocess_record` docstring and `compute_normalization_stats` /
    `apply_normalization_stats`, which should be fit on the training
    split only (after `patient_ids` is used to do a patient-level split).

    Parameters
    ----------
    records : list of PTBRecord
        Loaded PTB records.
    fs : int
        Sampling frequency.
    target_len : int, optional
        If given, resample each beat to this many points. If None
        (default), keep the raw fixed-window beat length.
    max_beats_per_record : int, optional
        If given, keep only the first N beats from each record.

    Returns
    -------
    dict with:
        'ecg': [Total_beats, 12, beat_len]
        'vcg': [Total_beats, 3, beat_len]
        'patient_ids': list of patient IDs per beat
        'beat_counts': number of beats per record
    """
    all_ecg = []
    all_vcg = []
    patient_ids = []
    beat_counts = []

    for rec in records:
        result = preprocess_record(
            rec.ecg, rec.vcg, fs=fs,
            target_len=target_len, max_beats=max_beats_per_record,
        )
        if result["n_beats"] > 0:
            all_ecg.append(result["ecg_beats"])
            all_vcg.append(result["vcg_beats"])
            patient_ids.extend([rec.patient_id] * result["n_beats"])
            beat_counts.append(result["n_beats"])

    if not all_ecg:
        return {
            "ecg": np.array([]),
            "vcg": np.array([]),
            "patient_ids": [],
            "beat_counts": [],
        }

    return {
        "ecg": np.concatenate(all_ecg, axis=0),
        "vcg": np.concatenate(all_vcg, axis=0),
        "patient_ids": patient_ids,
        "beat_counts": beat_counts,
    }


def preprocess_dataset_from_dir(
    data_dir,
    fs: int = 1000,
    target_len: int | None = None,
    max_beats_per_record: int | None = None,
    max_records: int | None = None,
    seed: int | None = 42,
) -> dict[str, np.ndarray]:
    """
    Load and preprocess PTB records one at a time, streaming through disk
    instead of materializing the whole dataset in memory first.

    `load_all_records()` + `preprocess_dataset()` holds every raw record
    (full-length, all 549 of them) in memory simultaneously *and* holds
    the growing set of extracted beats at the same time — on an 8GB WSL
    box this is enough to trigger the OOM killer once beat count per
    record goes up (confirmed via dmesg: pt_main_thread killed at
    anon-rss 7.6GB). This version discards each record's raw signal as
    soon as its beats are extracted, so peak memory is roughly
    "one raw record + all extracted beats so far" instead of
    "all raw records + all extracted beats".

    Parameters
    ----------
    data_dir : str or Path
        Directory containing PTB CSV files.
    fs : int
        Sampling frequency.
    target_len : int, optional
        If given, resample each beat to this many points. If None
        (default), keep the raw fixed-window beat length.
    max_beats_per_record : int, optional
        If given, keep only the first N beats from each record.
    max_records : int, optional
        Maximum number of records to load.
    seed : int, optional
        Seed for randomly sampling `max_beats_per_record` beats from each
        recording (rather than always taking the first N — see
        `preprocess_record`). Pass None to fall back to first-N selection.

    Returns
    -------
    dict with:
        'ecg': [Total_beats, 12, beat_len]
        'vcg': [Total_beats, 3, beat_len]
        'patient_ids': list of patient IDs per beat
        'beat_counts': number of beats per record
    """
    from vcg_project.data.ptb_loader import discover_records, load_record

    csv_paths = discover_records(data_dir)
    if max_records is not None:
        csv_paths = csv_paths[:max_records]

    rng = np.random.default_rng(seed) if seed is not None else None

    all_ecg = []
    all_vcg = []
    patient_ids = []
    beat_counts = []

    for path in csv_paths:
        try:
            rec = load_record(path, fs=fs)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")
            continue

        result = preprocess_record(
            rec.ecg, rec.vcg, fs=fs,
            target_len=target_len, max_beats=max_beats_per_record, rng=rng,
        )
        if result["n_beats"] > 0:
            all_ecg.append(result["ecg_beats"])
            all_vcg.append(result["vcg_beats"])
            patient_ids.extend([rec.patient_id] * result["n_beats"])
            beat_counts.append(result["n_beats"])

        # Drop the raw full-length record before moving to the next file —
        # this is the whole point of streaming instead of pre-loading.
        del rec

    if not all_ecg:
        return {
            "ecg": np.array([]),
            "vcg": np.array([]),
            "patient_ids": [],
            "beat_counts": [],
        }

    return {
        "ecg": np.concatenate(all_ecg, axis=0),
        "vcg": np.concatenate(all_vcg, axis=0),
        "patient_ids": patient_ids,
        "beat_counts": beat_counts,
    }


def split_by_patient(
    patient_ids: list[str],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split beat indices into train/val so that no patient appears in both.

    A random per-beat split leaks patient-specific characteristics
    (electrode placement, baseline morphology) across train/val, which
    inflates validation metrics without reflecting real generalization.
    Splitting by patient instead means val performance reflects the
    model's ability to generalize to unseen patients.

    Parameters
    ----------
    patient_ids : list of str
        One patient ID per beat (as returned by `preprocess_dataset`).
    val_fraction : float
        Fraction of *patients* (not beats) to hold out for validation.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    (train_idx, val_idx) : tuple of np.ndarray
        Beat indices for each split.
    """
    patient_ids = np.asarray(patient_ids)
    unique_patients = np.unique(patient_ids)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_patients)

    n_val_patients = max(1, int(len(unique_patients) * val_fraction))
    val_patients = set(shuffled[:n_val_patients])
    train_patients = set(shuffled[n_val_patients:])

    train_mask = np.isin(patient_ids, list(train_patients))
    val_mask = np.isin(patient_ids, list(val_patients))

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    return train_idx, val_idx
