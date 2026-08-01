"""
LSTM-based ECG → VCG converter.

Baseline model that learns a direct mapping from 12-lead ECG
to Frank XYZ VCG using convolutional feature extraction followed
by bidirectional LSTM temporal modeling.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ECGToVCG_LSTM(nn.Module):
    """
    LSTM encoder for ECG → VCG conversion.

    Architecture:
        Input: [B, 12, T] (12-lead ECG)
        → Conv1D feature extraction
        → Bidirectional LSTM
        → Linear projection to XYZ
        Output: [B, 3, T] (Frank XYZ VCG)

    Parameters
    ----------
    n_ecg_leads : int
        Number of input ECG leads (default 12).
    n_vcg_leads : int
        Number of output VCG leads (default 3).
    conv_channels : int
        Number of Conv1D filters.
    conv_kernel : int
        Conv1D kernel size.
    lstm_hidden : int
        LSTM hidden dimension.
    lstm_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_ecg_leads: int = 12,
        n_vcg_leads: int = 3,
        conv_channels: int = 64,
        conv_kernel: int = 7,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_ecg_leads = n_ecg_leads
        self.n_vcg_leads = n_vcg_leads

        # Conv1D feature extractor
        # Input: [B, 12, T] → [B, conv_channels, T']
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_ecg_leads, conv_channels, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
        )

        # Bidirectional LSTM
        # Input: [B, T', conv_channels] → [B, T', lstm_hidden * 2]
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Output projection: [B, T', lstm_hidden * 2] → [B, T', 3]
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, n_vcg_leads),
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
        # Conv feature extraction: [B, 12, T] → [B, C, T]
        features = self.conv_block(ecg)

        # LSTM expects [B, T, C], so transpose
        features = features.transpose(1, 2)  # [B, T, C]

        # LSTM: [B, T, C] → [B, T, hidden*2]
        lstm_out, _ = self.lstm(features)

        # Project to VCG: [B, T, hidden*2] → [B, T, 3]
        vcg = self.output_proj(lstm_out)

        # Transpose back to [B, 3, T]
        vcg = vcg.transpose(1, 2)

        return vcg

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
