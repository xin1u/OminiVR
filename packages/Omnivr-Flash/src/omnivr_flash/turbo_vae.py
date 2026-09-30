"""
UltraTinyVAE: 自包含的 encoder + decoder 蒸馏架构。
使用 pixel_unshuffle/pixel_shuffle 配合 TAEHV-style causal 网络。

适配高压缩比 VAE (spatial 16x, temporal 8x, latent 64ch):
  - patch_size=2 做 pixel_unshuffle/shuffle → 贡献 2x 空间
  - 内部 network 3 stage × spatial 2x = 8x → 总 spatial 16x
  - 内部 network 3 stage × temporal 2x = 8x → 总 temporal 8x
"""
from __future__ import annotations

from collections import namedtuple
from math import log2
from typing import Iterable, Literal, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

T = TypeVar("T")
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


# ---------------------------------------------------------------------------
# 基础组件 (来自 ultra_latent_up_v3，自包含避免循环依赖)
# ---------------------------------------------------------------------------

def _conv3x3(in_channels: int, out_channels: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs)


def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "silu":
        return nn.SiLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'.")


def _validate_power_of_two_scale(name: str, value: int) -> int:
    value = int(value)
    if value not in (1, 2, 4, 8):
        raise ValueError(f"{name} must be one of 1/2/4/8, got {value}.")
    return value


def _num_2x_stages(scale: int) -> int:
    return int(log2(_validate_power_of_two_scale("scale", scale)))


def _expand_stage_value(value, num_stages: int, name: str) -> list:
    if isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) != num_stages:
            raise ValueError(f"{name} must have length {num_stages}, got {len(values)}.")
        return values
    return [value for _ in range(num_stages)]


def _stage_factors_from_total_scale(
    total_scale: int, num_stages: int, name: str, align: Literal["left", "right"],
) -> list[int]:
    total_scale = _validate_power_of_two_scale(name, total_scale)
    num_2x = _num_2x_stages(total_scale)
    factors = [1] * num_stages
    if align == "left":
        indices = range(num_2x)
    else:
        indices = range(num_stages - num_2x, num_stages)
    for idx in indices:
        factors[idx] = 2
    return factors


def _validate_stage_factors(
    value: Iterable[int] | None, total_scale: int, num_stages: int,
    name: str, align: Literal["left", "right"],
) -> list[int]:
    if value is None:
        return _stage_factors_from_total_scale(total_scale, num_stages, name, align=align)
    factors = [int(v) for v in value]
    if len(factors) != num_stages:
        raise ValueError(f"{name}_stage_factors must have length {num_stages}, got {len(factors)}.")
    product = 1
    for v in factors:
        product *= v
    if product != int(total_scale):
        raise ValueError(f"Product of {name}_stage_factors must equal {total_scale}, got {factors}.")
    return factors


# ---------------------------------------------------------------------------
# CausalMemBlock
# ---------------------------------------------------------------------------

class CausalMemBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int | None = None,
        activation: str = "relu", memory_init: Literal["zero", "replicate"] = "replicate",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.memory_init = memory_init
        self.conv = nn.Sequential(
            _conv3x3(self.in_channels * 2, self.out_channels),
            _make_activation(activation),
            _conv3x3(self.out_channels, self.out_channels),
            _make_activation(activation),
            _conv3x3(self.out_channels, self.out_channels),
        )
        self.skip = (
            nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False)
            if self.in_channels != self.out_channels else nn.Identity()
        )
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], dim=1)) + self.skip(x))


# ---------------------------------------------------------------------------
# Encoder 组件: TPool + SpatialDownsample
# ---------------------------------------------------------------------------

class TPool(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.stride = int(stride)
        self.conv = nn.Conv2d(self.channels * self.stride, self.channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, c, h, w = x.shape
        return self.conv(x.reshape(-1, self.stride * c, h, w))


class SpatialDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Decoder 组件: TGrow + PixelShuffleUpsample
# ---------------------------------------------------------------------------

class TGrow(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.stride = int(stride)
        self.conv = nn.Conv2d(self.channels, self.channels * self.stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, c, h, w = x.shape
        return self.conv(x).reshape(nt * self.stride, c, h, w)


class PixelShuffleUpsample(nn.Module):
    def __init__(self, channels: int, scale_factor: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.scale_factor = int(scale_factor)
        if self.scale_factor == 1:
            self.block = nn.Identity()
        else:
            self.block = nn.Sequential(
                _conv3x3(self.channels, self.channels * self.scale_factor ** 2),
                nn.PixelShuffle(self.scale_factor),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Clamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


# ---------------------------------------------------------------------------
# Causal 推理引擎
# ---------------------------------------------------------------------------

def _apply_causal_encoder(model: nn.Sequential, x: torch.Tensor, parallel: bool) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected [B, C, T, H, W], got {tuple(x.shape)}.")
    b, c, t, h, w = x.shape
    if parallel:
        h_flat = rearrange(x, "b c t h w -> (b t) c h w")
        for block in model:
            if isinstance(block, CausalMemBlock):
                nt, cc, hh, ww = h_flat.shape
                tt = nt // b
                h_bt = h_flat.reshape(b, tt, cc, hh, ww)
                first = h_bt[:, :1] if block.memory_init == "replicate" else h_bt[:, :1] * 0
                past = torch.cat([first, h_bt[:, :-1]], dim=1).reshape(h_flat.shape)
                h_flat = block(h_flat, past)
            elif isinstance(block, TPool):
                if block.stride > 1:
                    nt, cc, hh, ww = h_flat.shape
                    tt = nt // b
                    pad_needed = (block.stride - tt % block.stride) % block.stride
                    if pad_needed > 0:
                        h_bt = h_flat.reshape(b, tt, cc, hh, ww)
                        h_bt = torch.cat([h_bt[:, :1].expand(-1, pad_needed, -1, -1, -1), h_bt], dim=1)
                        h_flat = h_bt.reshape(b * (tt + pad_needed), cc, hh, ww)
                    h_flat = block(h_flat)
                else:
                    h_flat = block(h_flat)
            else:
                h_flat = block(h_flat)
        nt, cc, hh, ww = h_flat.shape
        return rearrange(h_flat, "(b t) c h w -> b c t h w", b=b, t=nt // b)

    # sequential 模式：向前 pad 首帧使 T 整除总 temporal stride（和 parallel/teacher 行为一致）
    total_temporal_stride = 1
    for block in model:
        if isinstance(block, TPool) and block.stride > 1:
            total_temporal_stride *= block.stride
    pad_frames = (total_temporal_stride - t % total_temporal_stride) % total_temporal_stride
    if pad_frames > 0:
        x = torch.cat([x[:, :, :1].expand(-1, -1, pad_frames, -1, -1), x], dim=2)
        t = t + pad_frames

    caches = [None] * len(model)
    outputs: list[torch.Tensor] = []
    work_queue = [TWorkItem(x[:, :, i], 0) for i in range(t)]
    while work_queue:
        xt, block_idx = work_queue.pop(0)
        if block_idx == len(model):
            outputs.append(xt)
            continue
        block = model[block_idx]
        if isinstance(block, CausalMemBlock):
            past = caches[block_idx]
            if past is None:
                past = xt if block.memory_init == "replicate" else torch.zeros_like(xt)
            xt_next = block(xt, past.to(device=xt.device, dtype=xt.dtype))
            caches[block_idx] = xt.detach()
            work_queue.insert(0, TWorkItem(xt_next, block_idx + 1))
        elif isinstance(block, TPool):
            if caches[block_idx] is None:
                caches[block_idx] = []
            caches[block_idx].append(xt)
            if len(caches[block_idx]) == block.stride:
                n_batch, cc, hh, ww = xt.shape
                pooled = block(torch.cat(caches[block_idx], 1).view(n_batch * block.stride, cc, hh, ww))
                caches[block_idx] = []
                work_queue.insert(0, TWorkItem(pooled, block_idx + 1))
        else:
            work_queue.insert(0, TWorkItem(block(xt), block_idx + 1))
    return torch.stack(outputs, dim=2)


def _apply_causal_decoder(model: nn.Sequential, x: torch.Tensor, parallel: bool) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected [B, C, T, H, W], got {tuple(x.shape)}.")
    b, c, t, h, w = x.shape
    if parallel:
        h_flat = rearrange(x, "b c t h w -> (b t) c h w")
        for block in model:
            if isinstance(block, CausalMemBlock):
                nt, cc, hh, ww = h_flat.shape
                tt = nt // b
                h_bt = h_flat.reshape(b, tt, cc, hh, ww)
                first = h_bt[:, :1] if block.memory_init == "replicate" else h_bt[:, :1] * 0
                past = torch.cat([first, h_bt[:, :-1]], dim=1).reshape(h_flat.shape)
                h_flat = block(h_flat, past)
            elif isinstance(block, TGrow):
                h_flat = block(h_flat)
            else:
                h_flat = block(h_flat)
        nt, cc, hh, ww = h_flat.shape
        return rearrange(h_flat, "(b t) c h w -> b c t h w", b=b, t=nt // b)

    caches = [None] * len(model)
    outputs: list[torch.Tensor] = []
    work_queue = [TWorkItem(x[:, :, i], 0) for i in range(t)]
    while work_queue:
        xt, block_idx = work_queue.pop(0)
        if block_idx == len(model):
            outputs.append(xt)
            continue
        block = model[block_idx]
        if isinstance(block, CausalMemBlock):
            past = caches[block_idx]
            if past is None:
                past = xt if block.memory_init == "replicate" else torch.zeros_like(xt)
            xt_next = block(xt, past.to(device=xt.device, dtype=xt.dtype))
            caches[block_idx] = xt.detach()
            work_queue.insert(0, TWorkItem(xt_next, block_idx + 1))
        elif isinstance(block, TGrow):
            xt_next = block(xt)
            _, cc, hh, ww = xt_next.shape
            for frame in reversed(xt_next.view(b, block.stride * cc, hh, ww).chunk(block.stride, dim=1)):
                work_queue.insert(0, TWorkItem(frame, block_idx + 1))
        else:
            work_queue.insert(0, TWorkItem(block(xt), block_idx + 1))
    return torch.stack(outputs, dim=2)


# ---------------------------------------------------------------------------
# UltraTinyVAE: 统一的 encoder + decoder
# ---------------------------------------------------------------------------

class UltraTinyVAE(nn.Module):
    """
    causal tiny VAE (encoder + decoder).

    架构:
      Encoder: pixel_unshuffle(patch_size) → Conv → [TPool → SpatialDown → MemBlock ×N] ×3 → Conv
      Decoder: Clamp → Conv → [MemBlock ×N → PixelShuffleUp → TGrow → Conv] ×3 → Act → Conv → pixel_shuffle(patch_size)

    总压缩比 = patch_size × network_spatial_scale (空间), network_temporal_scale (时间)
    """

    def __init__(
        self,
        patch_size: int = 2,
        image_channels: int = 3,
        latent_channels: int = 64,
        encoder_channels: tuple[int, ...] | list[int] = (64, 64, 64),
        decoder_channels: tuple[int, ...] | list[int] = (512, 256, 128),
        blocks_per_stage: int | tuple[int, ...] | list[int] = 3,
        spatial_scale: int = 8,
        temporal_scale: int = 8,
        encoder_spatial_factors: list[int] | None = None,
        encoder_temporal_factors: list[int] | None = None,
        decoder_spatial_factors: list[int] | None = None,
        decoder_temporal_factors: list[int] | None = None,
        activation: str = "relu",
        encoder_memory_init: Literal["zero", "replicate"] = "replicate",
        decoder_memory_init: Literal["zero", "replicate"] = "zero",
        default_parallel: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.image_channels = int(image_channels)
        self.latent_channels = int(latent_channels)
        self.default_parallel = bool(default_parallel)
        num_stages = 3

        spatial_scale = _validate_power_of_two_scale("spatial_scale", spatial_scale)
        temporal_scale = _validate_power_of_two_scale("temporal_scale", temporal_scale)

        enc_spatial = _validate_stage_factors(encoder_spatial_factors, spatial_scale, num_stages, "enc_spatial", "left")
        enc_temporal = _validate_stage_factors(encoder_temporal_factors, temporal_scale, num_stages, "enc_temporal", "left")
        dec_spatial = _validate_stage_factors(decoder_spatial_factors, spatial_scale, num_stages, "dec_spatial", "left")
        dec_temporal = _validate_stage_factors(decoder_temporal_factors, temporal_scale, num_stages, "dec_temporal", "right")

        self.temporal_stage_factors = dec_temporal

        enc_ch = [int(c) for c in encoder_channels]
        dec_ch = [int(c) for c in decoder_channels]
        bps = [int(v) for v in _expand_stage_value(blocks_per_stage, num_stages, "blocks_per_stage")]

        # ---- Encoder ----
        enc_in = image_channels * patch_size ** 2
        enc_layers: list[nn.Module] = [_conv3x3(enc_in, enc_ch[0]), _make_activation(activation)]
        cur = enc_ch[0]
        for s in range(num_stages):
            enc_layers.append(TPool(cur, stride=enc_temporal[s]))
            if enc_spatial[s] == 2:
                enc_layers.append(SpatialDownsample(cur, cur))
            else:
                enc_layers.append(nn.Identity())
            for _ in range(bps[s]):
                enc_layers.append(CausalMemBlock(cur, cur, activation, encoder_memory_init))
            if s + 1 < num_stages:
                nxt = enc_ch[s + 1]
                enc_layers.append(_conv3x3(cur, nxt, bias=False))
                cur = nxt
        enc_layers.append(_conv3x3(cur, latent_channels))
        self.encoder_net = nn.Sequential(*enc_layers)

        # ---- Decoder ----
        dec_layers: list[nn.Module] = [Clamp(), _conv3x3(latent_channels, dec_ch[0]), _make_activation(activation)]
        cur = dec_ch[0]
        for s in range(num_stages):
            for _ in range(bps[s]):
                dec_layers.append(CausalMemBlock(cur, cur, activation, decoder_memory_init))
            dec_layers.append(PixelShuffleUpsample(cur, scale_factor=dec_spatial[s]))
            dec_layers.append(TGrow(cur, stride=dec_temporal[s]))
            if s + 1 < num_stages:
                nxt = dec_ch[s + 1]
                dec_layers.append(_conv3x3(cur, nxt, bias=False))
                cur = nxt
        dec_out = image_channels * patch_size ** 2
        dec_layers.append(_make_activation(activation))
        dec_layers.append(_conv3x3(cur, dec_out))
        self.decoder_net = nn.Sequential(*dec_layers)

    def encode(self, pixel: torch.Tensor, parallel: bool | None = None) -> torch.Tensor:
        """pixel [B, C, T, H, W] → latent [B, latent_ch, T', H', W']"""
        if parallel is None:
            parallel = self.default_parallel
        if self.patch_size > 1:
            b, c, t, h, w = pixel.shape
            x = rearrange(pixel, "b c t h w -> (b t) c h w")
            x = F.pixel_unshuffle(x, self.patch_size)
            x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
        else:
            x = pixel
        return _apply_causal_encoder(self.encoder_net, x, parallel)

    def decode(self, latent: torch.Tensor, parallel: bool | None = None) -> torch.Tensor:
        """latent [B, latent_ch, T', H', W'] → pixel [B, C, T, H, W]"""
        if parallel is None:
            parallel = self.default_parallel
        decoded = _apply_causal_decoder(self.decoder_net, latent, parallel)
        if self.patch_size > 1:
            b, c, t, h, w = decoded.shape
            x = rearrange(decoded, "b c t h w -> (b t) c h w")
            x = F.pixel_shuffle(x, self.patch_size)
            decoded = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
        return decoded
