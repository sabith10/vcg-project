"""
Transformer-based ECG → VCG converter with VCG-space bottleneck.

Inspired by the LVCG framework (Huang et al., ICML 2026). The key
innovation is an explicit VCG-space bottleneck that forces the model
to learn in physically grounded XYZ space rather than directly in
lead space.

Architecture:
    1. VCG lift: project 12-lead ECG → 3-lead VCG via pseudo-inverse
    2. Encode VCG waveforms with 1D convolutions
    3. Self-attention over time steps
    4. GRU models temporal dynamics
    5. Decode back to VCG waveforms
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard 12-lead geometry matrix: maps VCG [3] → ECG [12]
# E(t) = A @ V(t), where A is [12, 3]
# Source: LVCG paper Appendix A.3
LEAD_GEOMETRY_12 = torch.tensor([
    #     X       Y       Z
    [  0.386, -0.078,  0.108],  # I
    [  0.094,  0.938, -0.010],  # II
    [ -0.292,  0.860, -0.118],  # III
    [ -0.240,  0.430, -0.049],  # aVR
    [  0.246, -0.500,  0.118],  # aVL
    [ -0.058,  1.000, -0.108],  # aVF
    [ -0.510, -0.040, -0.610],  # V1
    [ -0.380, -0.040, -0.500],  # V2
    [ -0.100, -0.040, -0.270],  # V3
    [  0.180,  0.040,  0.040],  # V4
    [  0.410,  0.040,  0.320],  # V5
    [  0.570,  0.040,  0.500],  # V6
], dtype=torch.float32)  # [12, 3]


class VCGLift(nn.Module):
    """
    Non-learnable geometric transform: ECG leads → VCG XYZ.

    Implements Tikhonov-regularized pseudo-inverse of the lead
    geometry matrix A [12, 3]:

        V = (A^T A + εI)^{-1} A^T E

    This recovers the latent VCG from observed ECG leads.
    The transform is fixed (not learned), enforcing physical
    consistency with the dipole model.
    """

    def __init__(self, epsilon: float = 0.01):
        super().__init__()
        self.register_buffer("A", LEAD_GEOMETRY_12)  # [12, 3]

        # Precompute: (A^T A + εI)^{-1} A^T → [3, 12]
        ATA = self.A.T @ self.A  # [3, 3]
        I3 = torch.eye(3)
        # solve(ATA + εI, A^T) gives (ATA + εI)^{-1} A^T
        M = torch.linalg.solve(ATA + epsilon * I3, self.A.T)  # [3, 12]
        self.register_buffer("pseudo_inv", M)

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """
        Lift ECG [B, 12, T] → VCG [B, 3, T].
        """
        # einsum: pseudo_inv [3, 12] @ ecg [B, 12, T] → [B, 3, T]
        return torch.einsum("ij,bjt->bit", self.pseudo_inv, ecg)


class ECGToVCG_Transformer(nn.Module):
    """
    Transformer-based ECG → VCG converter.

    The model operates in two stages:
        1. Geometric lift: ECG → VCG via pseudo-inverse (non-learnable)
        2. Learned refinement: 1D conv + self-attention + GRU
           refines the geometrically-lifted VCG

    This two-stage design means the model starts from a physically
    grounded VCG estimate and learns to correct systematic errors
    in the geometric transform.

    Parameters
    ----------
    n_ecg_leads : int
        Number of input ECG leads (12).
    n_vcg_leads : int
        Number of output VCG leads (3).
    hidden_dim : int
        Internal hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer encoder layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_ecg_leads: int = 12,
        n_vcg_leads: int = 3,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_ecg_leads = n_ecg_leads
        self.n_vcg_leads = n_vcg_leads
        self.hidden_dim = hidden_dim

        # Non-learnable VCG lift
        self.vcg_lift = VCGLift(epsilon=0.01)

        # Encode the 3-lead VCG to hidden_dim
        self.vcg_encoder = nn.Sequential(
            nn.Conv1d(n_vcg_leads, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Also encode the original 12-lead ECG to get residuals
        self.ecg_encoder = nn.Sequential(
            nn.Conv1d(n_ecg_leads, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Combine VCG and ECG features
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Self-attention over time
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # GRU for temporal dynamics
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Decode back to VCG
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_vcg_leads),
        )

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        ecg : torch.Tensor
            Input ECG, shape [B, 12, T].

        Returns
        -------
        torch.Tensor
            Predicted VCG, shape [B, 3, T].
        """
        B, _, T = ecg.shape

        # Step 1: Geometric lift to VCG space
        vcg_init = self.vcg_lift(ecg)  # [B, 3, T]

        # Step 2: Encode both representations
        vcg_feat = self.vcg_encoder(vcg_init)    # [B, hidden, T]
        ecg_feat = self.ecg_encoder(ecg)         # [B, hidden, T]

        # Step 3: Combine features
        combined = torch.cat([vcg_feat, ecg_feat], dim=1)  # [B, hidden*2, T]
        combined = combined.transpose(1, 2)  # [B, T, hidden*2]
        combined = self.combine(combined)    # [B, T, hidden]

        # Step 4: Self-attention
        attended = self.attention(combined)  # [B, T, hidden]

        # Step 5: GRU
        gru_out, _ = self.gru(attended)  # [B, T, hidden]

        # Step 6: Decode to VCG
        vcg_pred = self.decoder(gru_out)  # [B, T, 3]

        return vcg_pred.transpose(1, 2)  # [B, 3, T]

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
