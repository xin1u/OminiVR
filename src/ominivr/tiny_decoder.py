"""Tiny AutoEncoder for LTX Video (TAEHV).

Lightweight decoder for fast latent → pixel preview during validation.
Operates on NTCHW latent tensors, outputs NTCHW RGB [0,1].

Architecture: 30.2M params (wide variant), 128-ch latent input, patch_size=4.

Adapted from Ollin Boer Bohan's Seraena/TAESD (MIT License), retrained for
LTX-2.3's 128-channel video latent space. See NOTICE for the full MIT license
text and attribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from tqdm.auto import tqdm

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class WideMemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        groups = max(1, n_out // 64)
        assert n_out % groups == 0, f"{n_out} % {groups} ??"
        self.conv = nn.Sequential(
            nn.Conv2d(n_in * 2, n_out, 1), nn.ReLU(inplace=True),
            conv(n_out, n_out, groups=groups), nn.ReLU(inplace=True),
            nn.Conv2d(n_out, n_out, 1), nn.ReLU(inplace=True),
            conv(n_out, n_out, groups=groups))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


def apply_model_with_memblocks_parallel(model, x, show_progress_bar):
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    x = x.reshape(N * T, C, H, W)

    for b in tqdm(model, disable=not show_progress_bar):
        if isinstance(b, (MemBlock, WideMemBlock)):
            NT, C, H, W = x.shape
            T = NT // N
            _x = x.reshape(N, T, C, H, W)
            block_memory = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
            x = b(x, block_memory)
        else:
            x = b(x)
    NT, C, H, W = x.shape
    T = NT // N
    return x.view(N, T, C, H, W)


def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    if parallel:
        return apply_model_with_memblocks_parallel(model, x, show_progress_bar)
    else:
        return _apply_sequential(model, x, show_progress_bar)


def _apply_sequential(model, x, show_progress_bar):
    assert x.ndim == 5
    work_queue = [TWorkItem(xt, 0) for xt in x.unbind(1)]
    memory = [None] * len(model)
    progress_bar = tqdm(range(len(work_queue)), disable=not show_progress_bar)
    out = []
    while work_queue:
        xt = _step(model, memory, work_queue, progress_bar)
        if xt is not None:
            out.append(xt)
    progress_bar.close()
    return torch.cat(out, 1)


def _step(model, memory, work_queue, progress_bar=None):
    while work_queue:
        xt, i = work_queue.pop(0)
        if progress_bar is not None and i == 0:
            progress_bar.update(1)
        if i == len(model):
            return xt.unsqueeze(1)
        b = model[i]
        if isinstance(b, (MemBlock, WideMemBlock)):
            if memory[i] is None:
                xt_new = b(xt, xt * 0)
            else:
                xt_new = b(xt, memory[i])
            memory[i] = xt
            work_queue.insert(0, TWorkItem(xt_new, i + 1))
        elif isinstance(b, TPool):
            if memory[i] is None:
                memory[i] = []
            memory[i].append(xt)
            if len(memory[i]) == b.stride:
                N, C, H, W = xt.shape
                xt = b(torch.cat(memory[i], 1).view(N * b.stride, C, H, W))
                memory[i] = []
                work_queue.insert(0, TWorkItem(xt, i + 1))
        elif isinstance(b, TGrow):
            xt = b(xt)
            NT, C, H, W = xt.shape
            for xt_next in reversed(xt.view(NT // b.stride, b.stride * C, H, W).chunk(b.stride, 1)):
                work_queue.insert(0, TWorkItem(xt_next, i + 1))
        else:
            xt = b(xt)
            work_queue.insert(0, TWorkItem(xt, i + 1))
    return None


class TAEHV(nn.Module):
    """Tiny AutoEncoder for LTX-2.3 Video latents.

    Input:  NTCHW latent, C=128, patch_size=4
    Output: NTCHW RGB [0,1]
    """

    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.patch_size = 4
        self.latent_channels = 128
        self.image_channels = 3

        n_f = [1024, 512, 256, 64]
        self.decoder = nn.Sequential(
            Clamp(), conv(self.latent_channels, n_f[0]), nn.ReLU(inplace=True),
            WideMemBlock(n_f[0], n_f[0]), WideMemBlock(n_f[0], n_f[0]), WideMemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2), TGrow(n_f[0], 2), conv(n_f[0], n_f[1], bias=False),
            WideMemBlock(n_f[1], n_f[1]), WideMemBlock(n_f[1], n_f[1]), WideMemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2), TGrow(n_f[1], 2), conv(n_f[1], n_f[2], bias=False),
            WideMemBlock(n_f[2], n_f[2]), WideMemBlock(n_f[2], n_f[2]), WideMemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2), TGrow(n_f[2], 2), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), conv(n_f[3], self.image_channels * self.patch_size ** 2),
        )

        self.t_upscale = 2 ** sum(t.stride == 2 for t in self.decoder if isinstance(t, TGrow))
        self.frames_to_trim = self.t_upscale - 1

        if checkpoint_path is not None:
            sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            # Original checkpoint has encoder + decoder; we only need decoder
            decoder_sd = {k: v for k, v in self._patch_tgrow(sd).items() if k.startswith("decoder.")}
            self.load_state_dict(decoder_sd, strict=True)

    def _patch_tgrow(self, sd):
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if key in sd and sd[key].shape[0] > new_sd[key].shape[0]:
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def decode_video(self, x, parallel=True, show_progress_bar=False):
        """Decode latent sequence to RGB video.

        Args:
            x: NTCHW latent tensor, C=128
            parallel: process all frames at once (fast, more memory)

        Returns:
            NTCHW RGB tensor in [0, 1]
        """
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        x = F.pixel_shuffle(x.flatten(0, 1), self.patch_size).unflatten(0, (x.shape[0], x.shape[1]))
        return x[:, self.frames_to_trim:].clamp_(0, 1)
