"""Strict model loading and one-time LoRA fusion for Omnivr-Flash."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from safetensors import safe_open
import torch

from . import logger
from .turbo_vae import UltraTinyVAE


DEFAULT_VAE_CONFIG = Path(__file__).resolve().parents[2] / "configs/turbo_vae.json"


@dataclass(frozen=True)
class ModelPaths:
    model: Path
    checkpoint: Path
    turbo_vae: Path
    prompt_cache: Path
    vae_config: Path = DEFAULT_VAE_CONFIG

    def validate(self) -> None:
        for name in ["model", "checkpoint", "turbo_vae", "prompt_cache", "vae_config"]:
            value = Path(getattr(self, name))
            if not value.is_file():
                raise FileNotFoundError(f"Missing {name}: {value}")
        if self.model.suffix != ".safetensors" or self.checkpoint.suffix != ".safetensors":
            raise ValueError("Base model and Flash LoRA must be safetensors files")


@dataclass(frozen=True)
class PromptEmbeddings:
    video_context_positive: torch.Tensor
    audio_context_positive: torch.Tensor


def load_prompt_cache(path: Path, device: torch.device) -> PromptEmbeddings:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict) or not isinstance(cache.get("val_positive"), dict):
        raise ValueError("Prompt cache must contain a val_positive dictionary")
    positive = cache["val_positive"]
    values = []
    for key, channels in [("video_encoding", 4096), ("audio_encoding", 2048)]:
        tensor = positive.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(f"Prompt cache needs a [1, tokens, {channels}] {key} tensor")
        if tensor.shape[0] != 1 or tensor.shape[1] == 0 or tensor.shape[2] != channels:
            raise ValueError(f"Invalid {key} shape: {tuple(tensor.shape)}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite {key} in prompt cache")
        values.append(tensor.to(device=device, dtype=torch.bfloat16))
    return PromptEmbeddings(*values)


@torch.inference_mode()
def merge_flash_checkpoint(model: torch.nn.Module, checkpoint: Path) -> dict:
    """Fuse W + B@A in FP32. Flash checkpoint B already includes alpha/rank.

    All LoRA pairs and all four additive condition tensors are required. A
    channel-concatenation OmniVR checkpoint is incompatible with this loader.
    """
    modules = dict(model.named_modules())
    parameters = dict(model.named_parameters())
    cond_names = {
        "_cond_video_proj.weight", "_cond_video_proj.bias",
        "_cond_audio_proj.weight", "_cond_audio_proj.bias",
    }
    required_cond = {f"diffusion_model.{name}" for name in cond_names}
    used = set()
    pairs = []
    ranks = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as state:
        keys = set(state.keys())
        missing = required_cond - keys
        if missing:
            raise ValueError(f"Not an additive-conditioning Omnivr-Flash checkpoint; missing {sorted(missing)}")
        # Validate the complete schema before changing any model weights.
        for key in sorted(keys):
            if not key.endswith(".lora_A.weight"):
                continue
            prefix = key.removesuffix(".lora_A.weight")
            name = prefix.removeprefix("diffusion_model.")
            b_key = prefix + ".lora_B.weight"
            if not isinstance(modules.get(name), torch.nn.Linear) or b_key not in keys:
                raise ValueError(f"Unmatched LoRA target or pair: {name}")
            a_shape = state.get_slice(key).get_shape()
            b_shape = state.get_slice(b_key).get_shape()
            weight = modules[name].weight
            if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
                raise ValueError(f"Invalid LoRA rank dimensions: {name}")
            if (b_shape[0], a_shape[1]) != tuple(weight.shape):
                raise ValueError(f"LoRA/base shape mismatch: {name}")
            pairs.append((name, key, b_key))
            ranks[str(a_shape[0])] = ranks.get(str(a_shape[0]), 0) + 1
            used.update([key, b_key])
        if not pairs:
            raise ValueError("Checkpoint has no LoRA tensor pairs")
        if keys - used != required_cond:
            raise ValueError(f"Unexpected checkpoint tensors: {sorted(keys - used - required_cond)}")
        for name in sorted(cond_names):
            key = f"diffusion_model.{name}"
            if name not in parameters or tuple(state.get_slice(key).get_shape()) != tuple(parameters[name].shape):
                raise ValueError(f"Condition projection mismatch: {name}")

        for name, a_key, b_key in pairs:
            weight = modules[name].weight
            a = state.get_tensor(a_key).to(device=weight.device, dtype=torch.float32)
            b = state.get_tensor(b_key).to(device=weight.device, dtype=torch.float32)
            weight.copy_((weight.float() + b @ a).to(weight.dtype))
            del a, b
        for name in sorted(cond_names):
            parameters[name].copy_(state.get_tensor(f"diffusion_model.{name}"))
    logger.info("Merged Flash LoRA and loaded additive video/audio conditioning")
    return {"lora_modules": len(pairs), "ranks": ranks, "condition_tensors": len(cond_names)}


def load_turbo_vae(paths: ModelPaths, device: torch.device) -> tuple[UltraTinyVAE, int]:
    config = json.loads(paths.vae_config.read_text())
    if not isinstance(config, dict):
        raise ValueError("Turbo VAE configuration must be a JSON object")
    model = UltraTinyVAE(**config)
    if config.get("latent_channels") != 128 or config.get("patch_size", 1) * config.get("spatial_scale", 1) != 32:
        raise ValueError("Flash requires 128 latent channels and total spatial compression of 32")
    if config.get("temporal_scale") != 8:
        raise ValueError("Flash requires temporal compression of 8")
    if paths.turbo_vae.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(paths.turbo_vae), device="cpu")
    else:
        state = torch.load(paths.turbo_vae, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    trim = math.prod(model.temporal_stage_factors) - 1
    logger.info("Loaded Turbo VAE encoder and decoder")
    return model, trim
