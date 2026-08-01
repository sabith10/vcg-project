"""
Kors regression transform for ECG → VCG conversion.

Classical linear transformation that maps 8 independent ECG leads
(I, II, V1-V6) to Frank XYZ VCG leads. No learning required —
just a matrix multiply.

Reference: Kors et al., "Reconstruction of the Frank vectorcardiogram
from standard electrocardiographic leads: diagnostic comparison of
different methods", European Heart Journal, 1990.
"""

from __future__ import annotations

import numpy as np


# Kors regression matrix: 3×8
# Maps [I, II, V1, V2, V3, V4, V5, V6] → [X, Y, Z]
# Source: Kors et al. 1990, Table 2
KORS_MATRIX = np.array([
    #  I      II     V1     V2     V3     V4     V5     V6
    [ 0.380,  0.136, -0.080, -0.108,  0.054,  0.239,  0.229,  0.184],  # X
    [-0.070,  0.936,  0.064, -0.010, -0.094, -0.010,  0.010, -0.010],  # Y
    [ 0.020, -0.016, -0.106, -0.112, -0.018,  0.048,  0.042, -0.066],  # Z
], dtype=np.float64)


# Indices of the 8 leads used by Kors in the standard 12-lead ordering:
# [I=0, II=1, V1=6, V2=7, V3=8, V4=9, V5=10, V6=11]
KORS_LEAD_INDICES = [0, 1, 6, 7, 8, 9, 10, 11]


# Also include the Inverse Dower Transform for comparison
# Source: Edenbrandt & Pahlm 1988, Guillem et al. 2006
INVERSE_DOWER_MATRIX = np.array([
    #  I      II     V1     V2     V3     V4     V5     V6
    [ 0.156, -0.010, -0.172, -0.074,  0.122,  0.231,  0.239,  0.194],  # X
    [-0.227,  0.887,  0.057, -0.019, -0.106, -0.022,  0.041,  0.048],  # Y
    [ 0.022,  0.102, -0.229, -0.310, -0.246, -0.063,  0.055,  0.108],  # Z
], dtype=np.float64)


def kors_transform(ecg_8leads: np.ndarray) -> np.ndarray:
    """
    Apply Kors regression to convert 8 ECG leads to XYZ VCG.

    Parameters
    ----------
    ecg_8leads : np.ndarray
        ECG signal, shape [8, T] where leads are
        [I, II, V1, V2, V3, V4, V5, V6].

    Returns
    -------
    np.ndarray
        VCG signal, shape [3, T] = [X, Y, Z].
    """
    return KORS_MATRIX @ ecg_8leads


def inverse_dower_transform(ecg_8leads: np.ndarray) -> np.ndarray:
    """
    Apply Inverse Dower Transform to convert 8 ECG leads to XYZ VCG.

    Parameters
    ----------
    ecg_8leads : np.ndarray
        ECG signal, shape [8, T].

    Returns
    -------
    np.ndarray
        VCG signal, shape [3, T].
    """
    return INVERSE_DOWER_MATRIX @ ecg_8leads


def kors_transform_beats(ecg_beats: np.ndarray) -> np.ndarray:
    """
    Apply Kors transform to batched beat data.

    Parameters
    ----------
    ecg_beats : np.ndarray
        Shape [N, 12, T] or [12, T]. If 12 leads, selects the 8 used by Kors.

    Returns
    -------
    np.ndarray
        Shape [N, 3, T] or [3, T].
    """
    if ecg_beats.ndim == 3:
        # Batched: [N, 12, T] → select 8 leads → [N, 8, T]
        ecg_8 = ecg_beats[:, KORS_LEAD_INDICES, :]
        # Apply transform: [3, 8] @ [N, 8, T] → [N, 3, T]
        return np.einsum("ij,njt->nit", KORS_MATRIX, ecg_8)
    elif ecg_beats.ndim == 2:
        # Single: [12, T] → select 8 leads → [8, T]
        ecg_8 = ecg_beats[KORS_LEAD_INDICES, :]
        return KORS_MATRIX @ ecg_8
    else:
        raise ValueError(f"Expected 2D or 3D input, got {ecg_beats.ndim}D")


def get_kors_lead_indices() -> list[int]:
    """Return the indices of the 8 leads used by Kors transform."""
    return KORS_LEAD_INDICES.copy()
