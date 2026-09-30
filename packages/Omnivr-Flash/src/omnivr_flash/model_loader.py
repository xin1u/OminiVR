"""Inference component loaders for the bundled LTX checkpoint format."""

from __future__ import annotations

from pathlib import Path

import torch

from ltx_core.loader import SingleGPUModelBuilder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoder,
    AudioDecoderConfigurator,
    AudioEncoder,
    AudioEncoderConfigurator,
    Vocoder,
    VocoderConfigurator,
)
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.model_configurator import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
)


def load_transformer(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> LTXModel:
    """Load the joint audio/video transformer from the base checkpoint."""
    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    ).build(device=torch.device(device), dtype=dtype)


def load_audio_vae_encoder(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> AudioEncoder:
    """Load the audio encoder used to prepare input conditioning."""
    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioEncoderConfigurator,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=torch.device(device), dtype=dtype)


def load_audio_vae_decoder(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> AudioDecoder:
    """Load the audio decoder used to restore the output spectrogram."""
    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioDecoderConfigurator,
        model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=torch.device(device), dtype=dtype)


def load_vocoder(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Vocoder:
    """Load the vocoder used to synthesize the restored audio waveform."""
    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VocoderConfigurator,
        model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
    ).build(device=torch.device(device), dtype=dtype)
