from __future__ import annotations

import torch
from torch import nn


def activation_layer(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation: {name}")


class MixerBlock(nn.Module):
    """TSMixer residual block adapted from local GCAD's ResBlock."""

    def __init__(self, history_length: int, channels: int, hidden_size: int, dropout: float, activation: str):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(channels)
        self.temporal_linear = nn.Linear(history_length, history_length)
        self.temporal_activation = activation_layer(activation)
        self.temporal_dropout = nn.Dropout(dropout)
        self.feature_norm = nn.LayerNorm(channels)
        self.feature_mlp = nn.Sequential(
            nn.Linear(channels, hidden_size),
            activation_layer(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal_norm(x).transpose(1, 2)
        temporal = self.temporal_dropout(self.temporal_activation(self.temporal_linear(temporal))).transpose(1, 2)
        residual = x + temporal
        return residual + self.feature_mlp(self.feature_norm(residual))


class SourceGCADMixer(nn.Module):
    """Multi-channel one-step predictor retaining temporal/feature mixing."""

    def __init__(
        self,
        history_length: int,
        channels: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.history_length = history_length
        self.channels = channels
        self.blocks = nn.ModuleList(
            [MixerBlock(history_length, channels, hidden_size, dropout, activation) for _ in range(num_layers)]
        )
        self.output_head = nn.Linear(history_length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.history_length, self.channels):
            raise ValueError(
                f"expected [batch,{self.history_length},{self.channels}], got {list(x.shape)}"
            )
        for block in self.blocks:
            x = block(x)
        return self.output_head(x.transpose(1, 2)).squeeze(-1)

    @staticmethod
    def per_channel_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return L[t,j] without reducing the output-channel dimension."""
        return nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")

