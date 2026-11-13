"""Compact residual MLP for one fixed horizon and fixed-rate causal history."""

import math

import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, history_frames, hidden_size=64, *, input_features="positions",
                 output_baseline="hold", horizon_steps=None):
        super().__init__()
        if type(history_frames) is not int or not 1 <= history_frames <= 1000:
            raise ValueError("history_frames must be between 1 and 1000")
        if type(hidden_size) is not int or not 1 <= hidden_size <= 1024:
            raise ValueError("hidden_size must be between 1 and 1024")
        self.history_frames = history_frames
        if input_features not in ("positions", "relative_last"):
            raise ValueError("unsupported predictor input_features")
        self.input_features = input_features
        if output_baseline not in ("hold", "constant_velocity"):
            raise ValueError("unsupported predictor output_baseline")
        if output_baseline == "constant_velocity":
            if (history_frames < 2 or type(horizon_steps) not in (int, float)
                    or not math.isfinite(horizon_steps) or horizon_steps <= 0):
                raise ValueError("constant_velocity requires two frames and positive horizon_steps")
        elif horizon_steps is not None:
            raise ValueError("hold does not use horizon_steps")
        self.output_baseline, self.horizon_steps = output_baseline, horizon_steps
        self.network = nn.Sequential(
            nn.Linear(history_frames * 48, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 48),
        )
        # Epoch zero is the declared analytic baseline; learn only its correction.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, history):
        if history.ndim != 4 or history.shape[1:] != (self.history_frames, 17, 3):
            raise ValueError("predictor expects (batch, history_frames, 17, 3)")
        features = history[:, :, 1:, :]
        if self.input_features == "relative_last":
            features = features - features[:, -1:, :, :]
        residual = self.network(features.flatten(1)).reshape(-1, 16, 3)
        baseline = history[:, -1, 1:, :]
        if self.output_baseline == "constant_velocity":
            baseline = baseline + (history[:, -1, 1:, :] - history[:, -2, 1:, :]) * (
                self.horizon_steps)
        result = baseline + residual
        return torch.cat((torch.zeros_like(result[:, :1, :]), result), dim=1)
