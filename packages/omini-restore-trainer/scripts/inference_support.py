#!/usr/bin/env python3
"""Inference support for OmniVR channel-concat conditioning and LoRA loading.

Expands the video/audio patchify projections and prepares concatenated
conditioning tokens to match the checkpoint. Also reads LoRA module ranks.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

logger = logging.getLogger("OmniVR")


def _read_lora_spec_from_checkpoint(ckpt_path: str) -> tuple[list[str], int, dict[str, int]]:
    """Read a LoRA checkpoint and return (target_module_names, default_rank, rank_pattern).

    Scans all lora_A tensors to discover module names and per-module ranks.
    Returns module names as full paths within the transformer (suitable for
    PEFT regex matching), the most common rank as default, and a rank_pattern
    dict for modules that differ.
    """
    sd = load_file(ckpt_path)
    module_ranks: dict[str, int] = {}
    for key, tensor in sd.items():
        if ".lora_A." not in key:
            continue
        # "diffusion_model.xxx.lora_A.weight" -> "xxx"
        name = key.replace("diffusion_model.", "", 1)
        name = re.sub(r"\.lora_A\.weight$", "", name)
        module_ranks[name] = tensor.shape[0]  # rank = lora_A rows

    # Most common rank -> default; others go into rank_pattern
    rank_counts = Counter(module_ranks.values())
    default_rank = rank_counts.most_common(1)[0][0]
    rank_pattern = {name: r for name, r in module_ranks.items() if r != default_rank}

    module_names = sorted(module_ranks.keys())
    logger.info(
        f"Checkpoint LoRA spec: {len(module_names)} modules, "
        f"default rank={default_rank}, {len(rank_pattern)} overrides"
    )
    return module_names, default_rank, rank_pattern


class ConcatConditioning:
    """Static helpers for AV restoration channel-concat condition injection.

    Expands patchify_proj from Linear(128, dim) to Linear(256, dim) with
    zero-initialized LQ channels, then patches the transformer's argument
    preprocessor to concatenate (rather than add) the LQ condition latent.
    """

    @staticmethod
    def _expand_patchify_proj(model) -> None:
        """Expand patchify_proj and audio_patchify_proj from 128->256 input channels.

        The first 128 columns retain pretrained weights; the new 128 columns
        (LQ condition) are zero-initialized so the model starts from baseline.
        """
        import torch.nn as nn

        # Video patchify_proj
        old = model.patchify_proj
        in_features = old.in_features
        new = nn.Linear(in_features * 2, old.out_features, bias=True)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :in_features].copy_(old.weight)
            new.bias.copy_(old.bias)
        model.patchify_proj = new
        logger.info(
            f"Expanded patchify_proj: Linear({in_features}, {old.out_features}) -> "
            f"Linear({in_features * 2}, {old.out_features}), LQ channels zero-init"
        )

        # Audio patchify_proj
        if hasattr(model, "audio_patchify_proj"):
            old_a = model.audio_patchify_proj
            in_a = old_a.in_features
            new_a = nn.Linear(in_a * 2, old_a.out_features, bias=True)
            with torch.no_grad():
                new_a.weight.zero_()
                new_a.weight[:, :in_a].copy_(old_a.weight)
                new_a.bias.copy_(old_a.bias)
            model.audio_patchify_proj = new_a
            logger.info(
                f"Expanded audio_patchify_proj: Linear({in_a}, {old_a.out_features}) -> "
                f"Linear({in_a * 2}, {old_a.out_features}), LQ channels zero-init"
            )

    @staticmethod
    def _patch_preprocessor_for_concat(model) -> None:
        """Monkey-patch TransformerArgsPreprocessor.prepare() to do channel-concat.

        Instead of:  x = patchify_proj(latent_128) + cond_proj(cond_latent_128)
        Do:          x = patchify_proj(cat([latent_128, cond_latent_128], dim=-1))

        This keeps Modality.latent at 128-dim (compatible with to_denoised/Euler step)
        while patchify_proj(256->dim) handles the concatenated input.
        """
        from ltx_core.model.transformer.transformer_args import TransformerArgsPreprocessor

        def _concat_prepare(self_prep, modality, cross_modality=None):
            if modality.cond_latent is not None:
                combined = torch.cat([modality.latent, modality.cond_latent], dim=-1)
                x = self_prep.patchify_proj(combined)
            else:
                pad = torch.zeros_like(modality.latent)
                combined = torch.cat([modality.latent, pad], dim=-1)
                x = self_prep.patchify_proj(combined)

            batch_size = x.shape[0]
            timestep, embedded_timestep = self_prep._prepare_timestep(
                modality.timesteps, self_prep.adaln, batch_size, modality.latent.dtype
            )
            prompt_timestep = None
            if self_prep.prompt_adaln is not None:
                prompt_timestep, _ = self_prep._prepare_timestep(
                    modality.sigma, self_prep.prompt_adaln, batch_size, modality.latent.dtype
                )
            context = self_prep._prepare_context(modality.context, x)
            attention_mask = self_prep._prepare_attention_mask(modality.context_mask, modality.latent.dtype)
            pe = self_prep._prepare_positional_embeddings(
                positions=modality.positions,
                inner_dim=self_prep.inner_dim,
                max_pos=self_prep.max_pos,
                use_middle_indices_grid=self_prep.use_middle_indices_grid,
                num_attention_heads=self_prep.num_attention_heads,
                x_dtype=modality.latent.dtype,
            )
            self_attention_mask = self_prep._prepare_self_attention_mask(modality.attention_mask, modality.latent.dtype)

            from ltx_core.model.transformer.transformer_args import TransformerArgs
            return TransformerArgs(
                x=x,
                context=context,
                context_mask=attention_mask,
                timesteps=timestep,
                embedded_timestep=embedded_timestep,
                positional_embeddings=pe,
                cross_positional_embeddings=None,
                cross_scale_shift_timestep=None,
                cross_gate_timestep=None,
                enabled=modality.enabled,
                prompt_timestep=prompt_timestep,
                self_attention_mask=self_attention_mask,
            )

        TransformerArgsPreprocessor.prepare = _concat_prepare
        logger.info("Patched TransformerArgsPreprocessor.prepare() for channel-concat injection")
