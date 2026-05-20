"""
Encodeur CNN pour cartes locales (6 canaux : 4 minéraux + obstacles + exploration).

CNN Q-Network — architecture from the article:

  "A multi-robot deep Q-learning framework for priority-based
   sanitization of railway stations" (Caccavale et al., 2023)
   DOI: 10.1007/s10489-023-04529-

Article CNN (Fig. 4), adapted to OBS_CHANNELS input channels
and 100×100 global observation:

    Conv2d(C → 32,  8×8, stride=4)  ReLU  →  (B, 32, 24, 24)
    Conv2d(32 → 64, 4×4, stride=2)  ReLU  →  (B, 64, 11, 11)
    Conv2d(64 → 64, 3×3, stride=1)  ReLU  →  (B, 64,  9,  9)
    Flatten                                →  (B, 5184)
    Linear(5184, 512)                ReLU  →  (B, 512)
    Linear(512,  num_actions)        —     →  (B, 8)

The robot position is encoded as a dedicated observation channel
(binary mask) so no separate position MLP is needed.

Dimension trace for H=W=100:
  floor((100-8)/4)+1 = 24   after Conv1
  floor((24-4)/2)+1  = 11   after Conv2
  11-3+1             =  9   after Conv3
  64 × 9 × 9         = 5184 after Flatten
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Article-conformant DQN for global 100×100 REE observations.

    Input:  (B, OBS_CHANNELS, 100, 100)  — channels defined in config.py
    Output: (B, num_actions)              — Q-value for each of the 8 actions

    forward(obs) → (B, num_actions)
    encode(obs)  → (B, 512)   [feature vector before Q-head, for analysis]
    """

    # Fixed conv-output size for H=W=100 — recomputed in __init__ if needed
    _CONV_FLAT: int = 64 * 9 * 9  # = 5184

    def __init__(
        self,
        obs_channels: int = 6,
        num_actions: int = 8,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # Dynamically compute flat size in case input dims differ from 100×100
        flat = self._get_flat_size(obs_channels)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 512),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(512, num_actions)

    def _get_flat_size(self, obs_channels: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, obs_channels, 100, 100)
            return self.conv(dummy).view(1, -1).size(1)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Return 512-d feature vector (before Q-head). For analysis."""
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        return self.fc(self.conv(obs))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute Q-values.

        Args:
            obs: (B, OBS_CHANNELS, 100, 100) or (OBS_CHANNELS, 100, 100)

        Returns:
            (B, num_actions) Q-values — linear activation (no softmax)
        """
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        return self.head(self.encode(obs))


def obs_to_tensor(
    obs: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Convert (H, W, C) float32 numpy to (1, C, H, W) float32 tensor."""
    return (
        torch.from_numpy(obs)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
