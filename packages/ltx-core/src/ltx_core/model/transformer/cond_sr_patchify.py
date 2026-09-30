"""Learned cross-resolution patchify projection for SR conditioning.

Maps LQ latent [B, C, T, H_lq, W_lq] → HQ token space [B, T*H_hq*W_hq, inner_dim]
via learned spatial mapping (Conv refinement + Linear spatial projection).

Unlike naive F.interpolate, uses trainable parameters to learn the optimal
spatial mapping from LQ grid to HQ grid in latent space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class CondSRPatchifyProj(nn.Module):
    """Learned cross-resolution patchify projection.

    Architecture:
      1. Conv2d residual block at LQ spatial resolution (per-frame feature refinement)
      2. Learned spatial mapping: Linear(H_lq*W_lq, H_hq*W_hq) — maps positions
      3. Channel projection: Linear(in_channels, inner_dim) — to transformer hidden dim

    Args:
        in_channels: Latent channel dim (128 for LTX VAE).
        inner_dim: Transformer hidden dim (4096 for LTX-2.3 22B video).
        lq_h: LQ latent height (H_lq = pixel_h / 32).
        lq_w: LQ latent width (W_lq = pixel_w / 32).
        hq_h: HQ latent height (H_hq = pixel_h / 32).
        hq_w: HQ latent width (W_hq = pixel_w / 32).
    """

    def __init__(
        self,
        in_channels: int,
        inner_dim: int,
        lq_h: int,
        lq_w: int,
        hq_h: int,
        hq_w: int,
    ) -> None:
        super().__init__()
        self.lq_h = lq_h
        self.lq_w = lq_w
        self.hq_h = hq_h
        self.hq_w = hq_w

        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

        lq_spatial = lq_h * lq_w
        hq_spatial = hq_h * hq_w
        self.spatial_proj = nn.Linear(lq_spatial, hq_spatial)

        self.proj = nn.Linear(in_channels, inner_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H_lq, W_lq] raw LQ latent (5D).
        Returns:
            [B, T*H_hq*W_hq, inner_dim] tokens in HQ spatial grid.
        """
        B, C, T, H, W = x.shape
        # Per-frame conv refinement at LQ resolution
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = x + self.pre_conv(x)
        # Flatten spatial → learned position mapping
        x = x.flatten(2)  # [(B*T), C, H_lq*W_lq]
        x = self.spatial_proj(x)  # [(B*T), C, H_hq*W_hq]
        # Reshape to token sequence
        x = rearrange(x, "(b t) c s -> b (t s) c", b=B, t=T)
        # Project to transformer hidden dim
        x = self.proj(x)  # [B, T*H_hq*W_hq, inner_dim]
        return x

    @staticmethod
    def init_near_identity(module: "CondSRPatchifyProj", proj_std: float = 1e-6) -> None:
        """Initialize for stable training start.

        - pre_conv: default init + residual means ~identity at start
        - spatial_proj: nearest-neighbor-like initialization
        - proj: near-zero so SR condition has minimal initial effect
        """
        # spatial_proj: init as approximate nearest-neighbor mapping
        with torch.no_grad():
            lq_h, lq_w = module.lq_h, module.lq_w
            hq_h, hq_w = module.hq_h, module.hq_w
            weight = torch.zeros(hq_h * hq_w, lq_h * lq_w)
            for hq_idx in range(hq_h * hq_w):
                hq_y = hq_idx // hq_w
                hq_x = hq_idx % hq_w
                lq_y = min(int(hq_y * lq_h / hq_h), lq_h - 1)
                lq_x = min(int(hq_x * lq_w / hq_w), lq_w - 1)
                lq_idx = lq_y * lq_w + lq_x
                weight[hq_idx, lq_idx] = 1.0
            module.spatial_proj.weight.copy_(weight)
            module.spatial_proj.bias.zero_()

        # proj: near-zero init
        nn.init.normal_(module.proj.weight, std=proj_std)
        nn.init.zeros_(module.proj.bias)
