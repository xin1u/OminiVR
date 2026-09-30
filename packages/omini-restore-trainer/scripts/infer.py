#!/usr/bin/env python3
# OmniVR release integration, 2026-09-30: bundled TinyDecoder and portable CLI paths.
"""OminiRestore Inference Script — AV Restoration with Trained LoRA.

Loads LQ videos from a folder, resizes to target resolution, runs restoration
with the trained concat-injection LoRA, and saves restored videos.

Three sampling strategies:
  A) no_cfg:     guidance_scale=1.0, single forward per step (fastest)
  B) empty_cfg:  positive=SR prompt, negative=empty string (matches training distribution)
  C) neg_cfg:    positive=SR prompt, negative=negative prompt (strongest enhancement)

Usage:
    python scripts/infer.py \\
        --input_dir /path/to/lq_videos \\
        --output_dir /path/to/output \\
        --checkpoint /path/to/lora_weights_step_XXXXX.safetensors \\
        --strategy neg_cfg \\
        --guidance_scale 5.0 \\
        --num_steps 15

    # Compare all 3 strategies on same inputs:
    python scripts/infer.py \\
        --input_dir /path/to/lq_videos \\
        --output_dir /path/to/output \\
        --checkpoint /path/to/lora.safetensors \\
        --strategy all
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _pkg in ("ltx-core", "ltx-pipelines"):
    _src = _REPO_ROOT / "packages" / _pkg / "src"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from inference_support import ConcatConditioning  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed prompts (same as training)
# ---------------------------------------------------------------------------

SR_FIXED_PROMPT = (
    "Extremely sharp and in focus, perfectly exposed with ideal brightness, high contrast, "
    "vibrant saturated colors, minimal noise, clean smooth texture, excellent natural lighting, "
    "stable without flickering, no motion blur, correct anatomical proportions, realistic natural "
    "skin tones, symmetrical well-defined facial features, correct number of limbs, properly formed "
    "hands with five fingers, clean readable text, consistent accurate perspective, perfectly stable "
    "camera, correct depth of field with natural bokeh, clean uncluttered background, no distracting "
    "reflections, soft natural shadows, consistent lighting direction, smooth color gradients without "
    "banding, photorealistic rendering, physically accurate materials, natural human appearance, "
    "correct ethnicity and gender representation, subtle natural expressions, accurate gaze direction, "
    "perfectly synchronized lip movements, clear natural voice audio, clean undistorted sound, no echo "
    "or background noise, perfectly synchronized audio and video, natural authentic dialogue, smooth "
    "fluid movement, natural pacing and timing, seamless transitions, consistent professional framing, "
    "level horizon, dimensional cinematic lighting, consistent visual tone, natural color grading, "
    "unfiltered organic look, and no digital artifacts."
)

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def tensor_debug_stats(name: str, tensor: torch.Tensor | None, max_items: int = 2_000_000):
    import os
    if os.environ.get("OMINIRESTORE_DEBUG_STATS", "0").lower() not in {"1", "true", "yes", "on"}:
        return
    if tensor is None:
        print(f"  [stats] {name}: None")
        return
    with torch.no_grad():
        x = tensor.detach()
        flat = x.reshape(-1)
        if flat.numel() > max_items:
            stride = max(1, flat.numel() // max_items)
            flat = flat[::stride]
        flat_f = flat.float()
        finite = torch.isfinite(flat_f)
        finite_ratio = finite.float().mean().item() if finite.numel() else 1.0
        if finite.any():
            vals = flat_f[finite]
            print(
                f"  [stats] {name}: shape={tuple(tensor.shape)} sample={flat_f.numel()} "
                f"finite={finite_ratio:.6f} mean={vals.mean().item():.6f} "
                f"std={vals.std(unbiased=False).item():.6f} min={vals.min().item():.6f} max={vals.max().item():.6f}"
            )
        else:
            print(f"  [stats] {name}: shape={tuple(tensor.shape)} sample={flat_f.numel()} finite=0.000000")


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def load_video(
    path: Path,
    target_h: int,
    target_w: int,
    max_frames: int = 121,
    resample_frames: int = 0,
    output_fps: float = 0.0,
) -> tuple[torch.Tensor, float, torch.Tensor | None, int]:
    """Load video aligned with the training RawAVDataset validation logic.

    Matches RawAVDataset._load_video: extract a clip of duration
    target_frames / CANONICAL_FPS (24.0), UNIFORMLY sampling `max_frames`
    frames across that whole window (not the first N frames), and report a
    canonical 24.0 fps. This keeps inference train-consistent — same frame
    count, same clip duration, same fps the model was trained/validated on —
    and fixes the "5s input -> 4s output" bug (old code took only the first
    121 source frames, dropping the tail).

    Returns:
        video: [1, 3, F, H, W] in [-1, 1], bf16
        fps: float (24.0 canonical)
        audio: [1, 2, T] in [-1, 1] or None
        audio_sr: int
    """
    import av
    import numpy as np

    CANONICAL_FPS = 24.0
    # target frame count must satisfy F % 8 == 1
    target_frames = ((max_frames - 1) // 8) * 8 + 1
    target_frames = max(target_frames, 1)
    clip_duration = float(target_frames) / CANONICAL_FPS

    # --- decode all frames (clip is ~5s, cheap), then uniformly sample ---
    container = av.open(str(path))
    src_fps = float(container.streams.video[0].average_rate)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
    container.close()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")

    total_frames = len(frames)
    total_duration = total_frames / max(src_fps, 1e-6)
    clip_dur = min(clip_duration, total_duration)
    clip_start = 0.0  # inference: deterministic start (from the beginning)

    start_frame = int(clip_start * src_fps)
    end_frame = min(start_frame + int(clip_dur * src_fps), total_frames)
    if end_frame - start_frame >= target_frames:
        # uniform sampling across the whole clip window (matches val linspace)
        idx = torch.linspace(start_frame, end_frame - 1, target_frames).long().tolist()
    else:
        idx = list(range(start_frame, end_frame))
    sel = [frames[i] for i in idx]

    video = torch.stack(sel)  # [F, 3, H, W]
    # pad by forward-reverse looping if the source is shorter than target
    if video.shape[0] < target_frames:
        segs, total = [video], video.shape[0]
        flip = True
        while total < target_frames:
            nxt = torch.flip(segs[-1], dims=[0]) if flip else segs[-1]
            segs.append(nxt); total += nxt.shape[0]; flip = not flip
        video = torch.cat(segs, dim=0)[:target_frames]
    valid_f = int(video.shape[0])
    effective_fps = CANONICAL_FPS

    # Resize to target resolution
    video = F.interpolate(video, size=(target_h, target_w), mode="bilinear", align_corners=False)
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, F, H, W]
    video = video * 2 - 1  # [0,1] → [-1,1]

    # --- audio: extract the same clip window [clip_start, clip_start+clip_dur] ---
    audio_data = None
    audio_sr = 44100
    try:
        container = av.open(str(path))
        if container.streams.audio:
            astream = container.streams.audio[0]
            audio_sr = astream.rate
            resampler = av.audio.resampler.AudioResampler(format="s16p", layout="stereo", rate=audio_sr)
            audio_frames = []
            for frame in container.decode(audio=0):
                for f in resampler.resample(frame):
                    audio_frames.append(torch.from_numpy(f.to_ndarray().copy()))
            if audio_frames:
                audio_data = torch.cat(audio_frames, dim=-1).float() / 32768.0  # [2, T_full]
                start_sample = int(clip_start * audio_sr)
                target_samples = int(clip_dur * audio_sr)
                audio_data = audio_data[..., start_sample:start_sample + target_samples]
                if audio_data.shape[-1] < target_samples:
                    audio_data = F.pad(audio_data, (0, target_samples - audio_data.shape[-1]))
                audio_data = audio_data.unsqueeze(0)  # [1, 2, T]
        container.close()
    except Exception:
        pass

    return video.to(torch.bfloat16), effective_fps, audio_data, audio_sr


def denoise_audio_waveform(
    wav: torch.Tensor,
    sr: int,
    highpass_hz: float = 200.0,
    specsub_over: float = 2.5,
    specsub_floor: float = 0.03,
) -> torch.Tensor:
    """Post-process audio: high-pass + spectral subtraction (matches the
    '4_hp200_specsub_strong' preset that sounded best on residual archival noise).

    The restored soundtrack keeps a low-frequency rumble (0-300 Hz, ~68% of the
    residual noise floor) plus a broadband hiss that the model does not fully
    remove. A high-pass cuts the rumble (little useful signal lives below 200 Hz),
    then spectral subtraction (over-subtraction from the quietest-10% noise
    estimate) removes the steady wideband floor. Applied per channel.

    wav: [C, T] or [T]. Returns same shape/dtype.
    """
    import numpy as np
    from scipy import signal as _sig

    single = wav.dim() == 1
    x = wav.detach().float().cpu().numpy()
    if single:
        x = x[None, :]
    out = np.empty_like(x)
    nyq = sr / 2.0
    sos = _sig.butter(4, min(highpass_hz / nyq, 0.99), "hp", output="sos") if highpass_hz > 0 else None
    for c in range(x.shape[0]):
        y = x[c]
        if sos is not None:
            y = _sig.sosfilt(sos, y)
        # spectral subtraction
        f, t, Z = _sig.stft(y, sr, nperseg=1024)
        mag, ph = np.abs(Z), np.angle(Z)
        e = (mag ** 2).sum(0)
        if e.size and np.isfinite(e).any():
            noise_mag = mag[:, e <= np.percentile(e, 10)].mean(1, keepdims=True)
            mag = np.maximum(mag - specsub_over * noise_mag, specsub_floor * mag)
        _, y = _sig.istft(mag * np.exp(1j * ph), sr, nperseg=1024)
        out[c, : min(len(y), out.shape[1])] = y[: out.shape[1]]
    out = np.clip(out, -1.0, 1.0)
    res = torch.from_numpy(out[0] if single else out).to(wav.dtype)
    return res


def save_video(video_01: torch.Tensor, path: Path, fps: float, audio: torch.Tensor | None = None, audio_sr: int = 44100):
    """Save [3, F, H, W] float [0,1] video + optional audio with exact frame count."""
    import subprocess
    import tempfile
    import shutil

    video_uint8 = (video_01.clamp(0, 1) * 255).byte().permute(1, 2, 3, 0).contiguous().cpu()  # [F, H, W, 3]
    num_frames, height, width, _ = video_uint8.shape

    def _write_video_only(dst: Path):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", f"{float(fps):.8f}",
            "-i", "-", "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18",
            str(dst),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            assert proc.stdin is not None
            for frame in video_uint8.numpy():
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            ret = proc.wait()
        finally:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        if ret != 0:
            raise RuntimeError(f"ffmpeg rawvideo encode failed with exit code {ret}: {dst}")

    if audio is None:
        _write_video_only(path)
        return

    with tempfile.TemporaryDirectory(dir=str(path.parent)) as tmpdir:
        tmp_video = Path(tmpdir) / "video.mp4"
        tmp_audio = Path(tmpdir) / "audio.wav"
        _write_video_only(tmp_video)
        try:
            import torchaudio
            audio_cpu = audio.float().cpu()
            if audio_cpu.dim() == 1:
                audio_cpu = audio_cpu.unsqueeze(0)
            target_audio_samples = int(round(num_frames / float(fps) * audio_sr))
            if audio_cpu.shape[-1] > target_audio_samples:
                audio_cpu = audio_cpu[..., :target_audio_samples]
            elif audio_cpu.shape[-1] < target_audio_samples:
                audio_cpu = F.pad(audio_cpu, (0, target_audio_samples - audio_cpu.shape[-1]))
            torchaudio.save(str(tmp_audio), audio_cpu, audio_sr)
        except Exception:
            shutil.move(str(tmp_video), str(path))
            return
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_video), "-i", str(tmp_audio),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-shortest", "-movflags", "+faststart", str(path)],
            check=True,
        )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    """Load base model + expand patchify + apply LoRA + load checkpoint."""
    from inference_model_loader import (
        load_model as load_ltx_model,
        load_embeddings_processor,
        load_text_encoder,
        load_video_vae_encoder,
    )

    device = torch.device(f"cuda:{args.gpu}")
    print(f"Loading base model from {args.model_path}...")
    components = load_ltx_model(
        checkpoint_path=args.model_path,
        device="cpu",
        dtype=torch.bfloat16,
        with_video_vae_encoder=True,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=args.with_audio,
        with_vocoder=args.with_audio,
        with_text_encoder=False,
    )

    transformer = components.transformer.to(dtype=torch.bfloat16)
    for frozen_module in (components.video_vae_decoder, components.audio_vae_decoder, components.vocoder):
        if frozen_module is not None:
            frozen_module.requires_grad_(False).eval()

    # Expand patchify_proj 128→256 + patch preprocessor for concat
    ConcatConditioning._expand_patchify_proj(transformer)
    ConcatConditioning._patch_preprocessor_for_concat(transformer)

    # Apply LoRA
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from inference_support import _read_lora_spec_from_checkpoint

    lora_structure_from = args.lora_structure or args.checkpoint
    module_names, default_rank, rank_pattern = _read_lora_spec_from_checkpoint(lora_structure_from)
    alpha_pattern = {k: v for k, v in rank_pattern.items()}

    lora_config = PeftLoraConfig(
        r=default_rank,
        lora_alpha=default_rank,
        target_modules=module_names,
        lora_dropout=0.0,
        init_lora_weights=True,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )
    transformer = get_peft_model(transformer, lora_config)

    # Rebuild preprocessors to reference LoRA-wrapped patchify_proj
    peft_base = transformer.get_base_model()
    peft_base._init_preprocessors(
        cross_pe_max_pos=max(
            peft_base.positional_embedding_max_pos[0],
            peft_base.audio_positional_embedding_max_pos[0],
        ) if hasattr(peft_base, "audio_positional_embedding_max_pos") else None
    )

    # Load checkpoint weights
    print(f"Loading LoRA checkpoint: {args.checkpoint}")
    state_dict = load_file(args.checkpoint)
    remapped = {}
    for k, v in state_dict.items():
        k = k.replace("diffusion_model.", "", 1)
        remapped[f"base_model.model.{k}"] = v
    set_peft_model_state_dict(transformer, remapped)

    # Merge LoRA into base weights → pure bf16, no fp32 PEFT overhead
    transformer = transformer.merge_and_unload()
    transformer = transformer.to(dtype=torch.bfloat16).eval()
    transformer.requires_grad_(False)
    print(f"LoRA merged into base weights (bf16, ~{sum(p.numel() for p in transformer.parameters()) / 1e9:.1f}B params)")
    # Keep on CPU initially — moved to GPU only during denoise

    # Load text embeddings — from cache if available, otherwise encode with Gemma
    prompt_cache = Path(args.prompt_cache) if args.prompt_cache else None

    if prompt_cache and prompt_cache.exists():
        print(f"Loading prompt embeddings from cache: {prompt_cache}")
        cached = torch.load(prompt_cache, map_location="cpu", weights_only=False)

        # val_positive/val_negative are Block 1+2+3 outputs (ready for inference)
        # empty: use val from empty_prompt processed through connector
        # For empty_cfg we need Block 1+2+3 of empty string — re-encode or approximate
        # The cache has sr_prompt (Block 1+2) and val_positive (Block 1+2+3).
        # For neg: val_negative is Block 1+2+3 of negative prompt.
        # For empty: we need to run empty through connector. Load connector only.
        from inference_model_loader import load_embeddings_processor
        from ltx_core.text_encoders.gemma.embeddings_processor import convert_to_additive_mask

        embeddings_processor = load_embeddings_processor(args.model_path, device=device, dtype=torch.bfloat16)
        embeddings_processor.requires_grad_(False).eval()

        # Process empty_prompt features through connector (Block 3)
        empty_feats = cached["empty_prompt"]
        with torch.no_grad():
            empty_v = empty_feats["video_feats"].to(device)
            empty_a = empty_feats["audio_feats"].to(device) if empty_feats["audio_feats"] is not None else None
            empty_mask = empty_feats["mask"].to(device)
            additive_mask = convert_to_additive_mask(empty_mask, empty_v.dtype)
            empty_v_enc, empty_a_enc, _ = embeddings_processor.create_embeddings(empty_v, empty_a, additive_mask)

        text_embeds = {
            "v_pos": cached["val_positive"]["video_encoding"].to(device),
            "a_pos": cached["val_positive"]["audio_encoding"].to(device),
            "v_neg": cached["val_negative"]["video_encoding"].to(device),
            "a_neg": cached["val_negative"]["audio_encoding"].to(device),
            "v_empty": empty_v_enc.to(device),
            "a_empty": empty_a_enc.to(device) if empty_a_enc is not None else None,
        }
        del embeddings_processor
        gc.collect()
        torch.cuda.empty_cache()
        print("Prompt embeddings loaded from cache (no Gemma needed)")

    else:
        print("Encoding text prompts with Gemma...")
        text_encoder = load_text_encoder(
            gemma_model_path=args.text_encoder_path,
            device=device,
            dtype=torch.bfloat16,
            load_in_8bit=False,
        )
        text_encoder.requires_grad_(False).eval()

        embeddings_processor = load_embeddings_processor(args.model_path, device=device, dtype=torch.bfloat16)
        embeddings_processor.requires_grad_(False).eval()

        with torch.no_grad():
            pos_out = embeddings_processor.process_hidden_states(
                *text_encoder.encode(SR_FIXED_PROMPT, padding_side="left"), padding_side="left"
            )
            neg_out = embeddings_processor.process_hidden_states(
                *text_encoder.encode(DEFAULT_NEGATIVE_PROMPT, padding_side="left"), padding_side="left"
            )
            empty_out = embeddings_processor.process_hidden_states(
                *text_encoder.encode("", padding_side="left"), padding_side="left"
            )

        text_embeds = {
            "v_pos": pos_out.video_encoding.to(device),
            "a_pos": pos_out.audio_encoding.to(device) if pos_out.audio_encoding is not None else None,
            "v_neg": neg_out.video_encoding.to(device),
            "a_neg": neg_out.audio_encoding.to(device) if neg_out.audio_encoding is not None else None,
            "v_empty": empty_out.video_encoding.to(device),
            "a_empty": empty_out.audio_encoding.to(device) if empty_out.audio_encoding is not None else None,
        }

        del text_encoder
        embeddings_processor.feature_extractor = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Text encoder unloaded")

    # VAE encoders
    vae_encoder = components.video_vae_encoder.to(device, torch.bfloat16).eval()
    vae_encoder.requires_grad_(False)

    audio_vae_encoder = None
    if args.with_audio:
        from inference_model_loader import load_audio_vae_encoder
        audio_vae_encoder = load_audio_vae_encoder(args.model_path, device=device, dtype=torch.bfloat16)
        audio_vae_encoder.requires_grad_(False).eval()

    # TinyDecoder (optional, ~38MB — avoids full VAE decoder OOM)
    tiny_decoder = None
    td_path = Path(args.tiny_decoder) if args.tiny_decoder else None
    if td_path and td_path.exists():
        from inference_tiny_decoder import TAEHV
        tiny_decoder = TAEHV(checkpoint_path=str(td_path)).to(device, torch.bfloat16).eval()
        print(f"TinyDecoder loaded: {td_path} (stays on GPU)")
    else:
        print(f"TinyDecoder not found at {td_path}, using full VAE decoder (needs offload)")

    # With TinyDecoder: keep transformer on GPU (total ~47GB)
    # Without: keep on CPU, offload during encode/decode
    if tiny_decoder is not None:
        transformer = transformer.to(device)

    return {
        "transformer": transformer,
        "vae_encoder": vae_encoder,
        "vae_decoder": components.video_vae_decoder,
        "audio_vae": components.audio_vae_decoder,
        "vocoder": components.vocoder,
        "audio_vae_encoder": audio_vae_encoder,
        "tiny_decoder": tiny_decoder,
        "text_embeds": text_embeds,
        "device": device,
        "decode_tile_size": args.decode_tile_size,
        "decode_tile_overlap": args.decode_tile_overlap,
        "decode_temporal_tile_frames": args.decode_temporal_tile_frames,
        "decode_temporal_overlap_frames": args.decode_temporal_overlap_frames,
    }


# ---------------------------------------------------------------------------
# Restoration inference
# ---------------------------------------------------------------------------

def restore_video(
    model_dict: dict,
    lq_video: torch.Tensor,       # [1, 3, F, H, W] bf16 [-1,1]
    lq_audio: torch.Tensor | None,  # [1, 2, T] or None
    audio_sr: int,
    fps: float,
    strategy: str,                 # "no_cfg" | "empty_cfg" | "neg_cfg"
    guidance_scale: float,
    num_steps: int,
    condition_noise: float,
    seed: int,
    target_frames: int | None = None,
    noise_shift: float | None = None,
    audio_denoise: bool = False,
    audio_denoise_highpass: float = 200.0,
    audio_denoise_over: float = 2.5,
    audio_denoise_floor: float = 0.03,
    lq_grayscale: bool = False,
    lq_gray_strength: float = 1.0,
    stg_scale: float = 0.0,
    stg_blocks: list[int] | None = None,
    stg_mode: str = "stg_av",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run restoration denoising. Returns (video_01 [3, F, H, W], audio_wav or None)."""

    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.guiders import CFGGuider, STGGuider
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.patchifiers import VideoLatentPatchifier, AudioPatchifier
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.guidance.perturbations import (
        BatchedPerturbationConfig, Perturbation, PerturbationConfig, PerturbationType,
    )
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.model import X0Model
    from ltx_core.model.video_vae import TilingConfig
    from ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig
    from ltx_core.tools import VideoLatentTools, AudioLatentTools
    from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape

    device = model_dict["device"]
    scale = SpatioTemporalScaleFactors.default()
    te = model_dict["text_embeds"]

    def _decode_tiling_config() -> TilingConfig:
        spatial_tile = int(model_dict.get("decode_tile_size", 512))
        spatial_overlap = int(model_dict.get("decode_tile_overlap", 64))
        temporal_tile = int(model_dict.get("decode_temporal_tile_frames", 64))
        temporal_overlap = int(model_dict.get("decode_temporal_overlap_frames", 24))
        if spatial_tile <= 0 or temporal_tile <= 0:
            return TilingConfig.default()
        return TilingConfig(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=spatial_tile,
                tile_overlap_in_pixels=spatial_overlap,
            ),
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=temporal_tile,
                tile_overlap_in_frames=temporal_overlap,
            ),
        )

    # Select text embeddings based on strategy
    v_pos = te["v_pos"]
    a_pos = te["a_pos"]
    if strategy == "no_cfg":
        v_neg, a_neg = None, None
        guidance_scale = 1.0
    elif strategy == "empty_cfg":
        v_neg, a_neg = te["v_empty"], te["a_empty"]
    else:  # neg_cfg
        v_neg, a_neg = te["v_neg"], te["a_neg"]

    # Encode LQ video
    has_td = model_dict.get("tiny_decoder") is not None
    if not has_td:
        model_dict["transformer"].to("cpu")
        torch.cuda.empty_cache()

    model_dict["vae_encoder"].to(device)
    # Train-test consistency: training degrades ~50% of clips to B&W and validation
    # forces LQ to grayscale, so the model expects grayscale LQ. Match that here.
    if lq_grayscale:
        g = (lq_video[:, 0:1] * 0.2989 + lq_video[:, 1:2] * 0.5870 + lq_video[:, 2:3] * 0.1140)
        gray = g.expand_as(lq_video)
        # Heavy desaturation but keep a little of the LQ color (s=1.0 full gray).
        s = float(lq_gray_strength)
        lq_video = (s * gray + (1.0 - s) * lq_video).contiguous()
    with torch.no_grad():
        lq_latent = model_dict["vae_encoder"].tiled_encode(lq_video.to(device), TilingConfig.default())
    model_dict["vae_encoder"].to("cpu")
    torch.cuda.empty_cache()

    _, _, num_frames, lat_h, lat_w = lq_latent.shape
    pixel_h = lat_h * scale.height
    pixel_w = lat_w * scale.width
    cond_pixel_f = (num_frames - 1) * scale.time + 1
    pixel_f = cond_pixel_f
    if target_frames is not None and target_frames > 0:
        if (target_frames - 1) % scale.time != 0:
            raise ValueError(f"target_frames must satisfy (F-1) % {scale.time} == 0, got {target_frames}")
        pixel_f = int(target_frames)
        print(f"  V2V target frames: {pixel_f} (condition frames: {cond_pixel_f})")

    video_patchifier = VideoLatentPatchifier(patch_size=1)
    audio_patchifier = AudioPatchifier(patch_size=1)

    lq_cond = video_patchifier.patchify(lq_latent)

    # Encode LQ audio
    lq_audio_cond = None
    if lq_audio is not None and model_dict["audio_vae_encoder"] is not None:
        from ltx_core.model.audio_vae import encode_audio
        from ltx_core.types import Audio
        audio_in = lq_audio.to(device, torch.bfloat16)
        if audio_in.shape[1] == 1:
            audio_in = audio_in.expand(-1, 2, -1)
        with torch.no_grad():
            audio_lat = encode_audio(Audio(waveform=audio_in, sampling_rate=audio_sr), model_dict["audio_vae_encoder"])
        lq_audio_cond = audio_patchifier.patchify(audio_lat)

    # Setup tools
    pixel_shape = VideoPixelShape(batch=1, frames=pixel_f, height=pixel_h, width=pixel_w, fps=fps)
    video_tools = VideoLatentTools(
        patchifier=video_patchifier,
        target_shape=VideoLatentShape.from_pixel_shape(pixel_shape),
        fps=fps, scale_factors=scale, causal_fix=True,
    )

    audio_tools = None
    audio_state = None
    if lq_audio_cond is not None:
        audio_tools = AudioLatentTools(
            patchifier=audio_patchifier,
            target_shape=AudioLatentShape.from_duration(batch=1, duration=float(pixel_f) / fps),
        )

    # Initial noise
    generator = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator)
    video_state = video_tools.create_initial_state(device, torch.bfloat16)
    video_state = noiser(video_state, noise_scale=1.0)

    if audio_tools is not None:
        audio_state = audio_tools.create_initial_state(device, torch.bfloat16)
        audio_state = noiser(audio_state, noise_scale=1.0)

    # Sigma schedule
    scheduler = LTX2Scheduler()
    target_shape = video_tools.target_shape
    tokens = target_shape.frames * target_shape.height * target_shape.width
    if num_steps == 1:
        sigmas = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)
    else:
        if noise_shift is not None and noise_shift > 0:
            max_shift = float(noise_shift)
        else:
            SHIFT_CAP = 13.0
            _x1, _x2, _base, _max = 1024, 4096, 0.95, 2.05
            raw_shift = (_max - _base) * (tokens - _x1) / (_x2 - _x1) + _base
            max_shift = _base + (SHIFT_CAP - _base) * (_x2 - _x1) / (tokens - _x1) if raw_shift > SHIFT_CAP else _max
        print(f"  Scheduler max_shift={max_shift:.4f} (tokens={tokens})")
        dummy = torch.empty(1, 1, target_shape.frames, target_shape.height, target_shape.width, device=device)
        sigmas = scheduler.execute(steps=num_steps, latent=dummy, max_shift=max_shift).to(device).float()

    stepper = EulerDiffusionStep()
    x0_model = X0Model(model_dict["transformer"])
    cfg_guider = CFGGuider(guidance_scale)
    # STG (spatio-temporal guidance) — matches training validation (stg_av).
    stg_guider = STGGuider(stg_scale)
    stg_ptb_config = None
    if stg_guider.enabled() and stg_blocks:
        _ptbs = [Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=stg_blocks)]
        if stg_mode == "stg_av":
            _ptbs.append(Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=stg_blocks))
        stg_ptb_config = BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=_ptbs)])
    video_mod = audio_mod = video_neg = audio_neg = None
    denoised_video = denoised_audio = pos_video = neg_video = neg_audio = None

    # Align LQ tokens
    def _align(tokens, target_len):
        if tokens is None:
            return None
        if tokens.shape[1] == target_len:
            return tokens
        if tokens.shape[1] > target_len:
            return tokens[:, :target_len]
        pad = torch.zeros(tokens.shape[0], target_len - tokens.shape[1], tokens.shape[2], device=device, dtype=tokens.dtype)
        return torch.cat([tokens, pad], dim=1)

    aligned_lq = _align(lq_cond, video_state.latent.shape[1])
    aligned_lq_audio = _align(lq_audio_cond, audio_state.latent.shape[1]) if audio_state is not None else None
    tensor_debug_stats("aligned_lq_condition", aligned_lq)
    tensor_debug_stats("initial_video_state_after_noise", video_state.latent)

    # Condition noise
    if condition_noise > 0:
        if aligned_lq is not None:
            aligned_lq = aligned_lq + torch.randn_like(aligned_lq) * condition_noise
        if aligned_lq_audio is not None:
            aligned_lq_audio = aligned_lq_audio + torch.randn_like(aligned_lq_audio) * condition_noise
    tensor_debug_stats("aligned_lq_after_condition_noise", aligned_lq)

    # Denoise
    if not has_td:
        model_dict["transformer"].to(device)
    print(f"  Denoising ({num_steps} steps, strategy={strategy}, cfg={guidance_scale})...")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for step_idx, sigma in enumerate(sigmas[:-1]):
            video_mod = Modality(
                enabled=True, latent=video_state.latent,
                sigma=sigma.repeat(video_state.latent.shape[0]),
                timesteps=sigma * video_state.denoise_mask,
                positions=video_state.positions,
                context=v_pos, context_mask=None, cond_latent=aligned_lq,
            )
            audio_mod = None
            if audio_state is not None:
                audio_mod = Modality(
                    enabled=True, latent=audio_state.latent,
                    sigma=sigma.repeat(audio_state.latent.shape[0]),
                    timesteps=sigma * audio_state.denoise_mask,
                    positions=audio_state.positions,
                    context=a_pos, context_mask=None, cond_latent=aligned_lq_audio,
                )

            denoised_video, denoised_audio = x0_model(video=video_mod, audio=audio_mod, perturbations=None)
            pos_video, pos_audio = denoised_video, denoised_audio

            # CFG
            if cfg_guider.enabled() and v_neg is not None:
                video_neg = replace(video_mod, context=v_neg)
                audio_neg = replace(audio_mod, context=a_neg) if audio_mod is not None else None
                neg_video, neg_audio = x0_model(video=video_neg, audio=audio_neg, perturbations=None)
                denoised_video = denoised_video + cfg_guider.delta(pos_video, neg_video)
                if audio_mod is not None and denoised_audio is not None and neg_audio is not None:
                    denoised_audio = denoised_audio + cfg_guider.delta(pos_audio, neg_audio)

            # STG (spatio-temporal guidance) — matches training validation
            if stg_guider.enabled() and stg_ptb_config is not None:
                ptb_video, ptb_audio = x0_model(video=video_mod, audio=audio_mod, perturbations=stg_ptb_config)
                denoised_video = denoised_video + stg_guider.delta(pos_video, ptb_video)
                if audio_mod is not None and denoised_audio is not None and ptb_audio is not None:
                    denoised_audio = denoised_audio + stg_guider.delta(pos_audio, ptb_audio)

            # Euler step
            video_state = replace(video_state, latent=stepper.step(
                sample=video_state.latent, denoised_sample=denoised_video,
                sigmas=sigmas, step_index=step_idx,
            ))
            if audio_state is not None and denoised_audio is not None:
                audio_state = replace(audio_state, latent=stepper.step(
                    sample=audio_state.latent, denoised_sample=denoised_audio,
                    sigmas=sigmas, step_index=step_idx,
                ))
            if step_idx == 0 or step_idx == len(sigmas) - 2 or (step_idx + 1) % 10 == 0:
                tensor_debug_stats(f"video_latent_after_step_{step_idx + 1}", video_state.latent)

    # Decode video
    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)
    tensor_debug_stats("video_latent_unpatchified_pre_decode", video_state.latent)

    td = model_dict.get("tiny_decoder")
    if td is not None:
        # TinyDecoder: fast, stays on GPU alongside transformer
        lat = video_state.latent.unsqueeze(0) if video_state.latent.dim() == 4 else video_state.latent
        vid_01 = td.decode_video(lat.permute(0, 2, 1, 3, 4).to(torch.bfloat16), parallel=True, show_progress_bar=False)
        video_01 = vid_01[0]  # [F, 3, H, W] in [0, 1]
        video_01 = video_01.permute(1, 0, 2, 3)  # [3, F, H, W]
    else:
        # Full VAE decoder: match trainer validation memory management.
        restored_latent = video_state.latent.detach().cpu()
        restored_audio_latent = None
        if audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            restored_audio_latent = audio_state.latent.detach().cpu()

        model_dict["transformer"].to("cpu")
        del x0_model, video_state, audio_state, video_tools, audio_tools
        del scheduler, stepper, noiser, sigmas, cfg_guider
        del lq_latent, lq_cond, aligned_lq, lq_audio_cond, aligned_lq_audio
        del video_mod, audio_mod, video_neg, audio_neg
        del denoised_video, denoised_audio, pos_video, neg_video, neg_audio
        gc.collect()
        torch.cuda.empty_cache()

        model_dict["vae_decoder"].to(device)
        decode_tiling = _decode_tiling_config()
        latent_for_decode = restored_latent.to(device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            video_chunks = list(model_dict["vae_decoder"].tiled_decode(latent_for_decode, decode_tiling))
        video_out = torch.cat(video_chunks, dim=2)
        tensor_debug_stats("video_out_raw_decode", video_out)
        model_dict["vae_decoder"].to("cpu")
        video_01 = video_out.squeeze(0).float().clamp(-1, 1) * 0.5 + 0.5
        tensor_debug_stats("video_01_after_decode", video_01)
        del latent_for_decode, restored_latent, video_chunks, video_out
        torch.cuda.empty_cache()

    # Decode audio
    audio_wav = None
    if td is None:
        audio_latent_for_decode = restored_audio_latent
    elif audio_state is not None and audio_tools is not None:
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)
        audio_latent_for_decode = audio_state.latent
    else:
        audio_latent_for_decode = None

    if audio_latent_for_decode is not None and model_dict["audio_vae"] is not None:
        model_dict["audio_vae"].to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            audio_mel = model_dict["audio_vae"](audio_latent_for_decode.to(device))
        model_dict["audio_vae"].to("cpu")
        if model_dict["vocoder"] is not None:
            model_dict["vocoder"].to(device)
            with torch.no_grad():
                audio_wav = model_dict["vocoder"](audio_mel).squeeze(0).float().cpu()
            model_dict["vocoder"].to("cpu")
            if audio_denoise and audio_wav is not None:
                voc_sr = getattr(model_dict["vocoder"], "output_sampling_rate", 44100)
                audio_wav = denoise_audio_waveform(
                    audio_wav, int(voc_sr),
                    highpass_hz=audio_denoise_highpass,
                    specsub_over=audio_denoise_over,
                    specsub_floor=audio_denoise_floor,
                )

    torch.cuda.empty_cache()
    return video_01.cpu(), audio_wav


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OminiRestore AV Inference")
    # Paths
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of LQ input videos (.mp4)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for restored videos")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA checkpoint (.safetensors)")
    parser.add_argument("--model_path", type=str, default=os.environ.get("OMINIVR_MODEL_PATH"),
                        required=not bool(os.environ.get("OMINIVR_MODEL_PATH")),
                        help="Path to base LTX-2.3 checkpoint, e.g. ltx-2.3-22b-dev.safetensors")
    parser.add_argument("--text_encoder_path", type=str, default=os.environ.get("OMINIVR_TEXT_ENCODER_PATH"),
                        help="Gemma text encoder directory; required when no prompt cache is available")
    parser.add_argument("--lora_structure", type=str, default=os.environ.get("OMINIVR_LORA_STRUCTURE"),
                        help="LoRA structure reference checkpoint (defaults to --checkpoint if unset)")
    parser.add_argument("--prompt_cache", type=str, default="ckpt/prompt/v5_sr_prompt_embeddings.pt",
                        help="Path to cached prompt embeddings (.pt). Skips Gemma loading if exists.")
    parser.add_argument("--tiny_decoder", type=str, default=None,
                        help="Optional legacy TAEHV weights; default uses the full LTX video VAE")

    # Sampling
    parser.add_argument("--strategy", type=str, default="neg_cfg", choices=["no_cfg", "empty_cfg", "neg_cfg", "all"],
                        help="CFG strategy: no_cfg (scale=1), empty_cfg (empty negative), neg_cfg (negative prompt), all (run all 3)")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG guidance scale (ignored for no_cfg)")
    parser.add_argument("--num_steps", type=int, default=15, help="Number of denoising steps")
    parser.add_argument("--condition_noise", type=float, default=0.3, help="Noise added to LQ condition (0=clean)")
    parser.add_argument("--noise_shift", type=float, default=0.0, help="Override LTX2Scheduler max_shift; <=0 uses automatic token-based shift")
    parser.add_argument("--seed", type=int, default=42)

    # Resolution
    parser.add_argument("--width", type=int, default=1920, help="Target width (must be divisible by 32)")
    parser.add_argument("--height", type=int, default=1088, help="Target height (must be divisible by 32)")
    parser.add_argument("--max_frames", type=int, default=57, help="Max frames (must satisfy F%%8==1). 57≈2.3s, 121≈4.8s")
    parser.add_argument("--target_frames", type=int, default=0, help="Optional generation target frames for v2v extrapolation; must satisfy F%%8==1")
    parser.add_argument("--resample_condition_frames", type=int, default=0,
                        help="Temporally resample decoded condition video to this frame count before VAE encode; <=0 keeps source cadence")
    parser.add_argument("--frame_rate", type=float, default=0.0, help="Override input fps for sampling/saving; <=0 keeps source fps")
    parser.add_argument("--decode_tile_size", type=int, default=512,
                        help="Full VAE decode spatial tile size in pixels; <=0 uses TilingConfig.default()")
    parser.add_argument("--decode_tile_overlap", type=int, default=64,
                        help="Full VAE decode spatial tile overlap in pixels")
    parser.add_argument("--decode_temporal_tile_frames", type=int, default=64,
                        help="Full VAE decode temporal tile size in frames; <=0 uses TilingConfig.default()")
    parser.add_argument("--decode_temporal_overlap_frames", type=int, default=24,
                        help="Full VAE decode temporal tile overlap in frames")

    # Device
    parser.add_argument("--gpu", type=int, default=0, help="GPU id for single-GPU mode")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for data-parallel inference (each GPU handles different videos)")
    parser.add_argument("--with_audio", action="store_true", default=True, help="Enable audio restoration")
    parser.add_argument("--no_audio", dest="with_audio", action="store_false")
    parser.add_argument("--audio_denoise", action="store_true", default=False,
                        help="Post-process restored audio: high-pass + spectral subtraction "
                             "(removes residual rumble + hiss the model leaves in).")
    parser.add_argument("--audio_denoise_highpass", type=float, default=200.0,
                        help="High-pass cutoff (Hz) for --audio_denoise; 0 disables the HPF stage.")
    parser.add_argument("--audio_denoise_over", type=float, default=2.5,
                        help="Spectral-subtraction over-subtraction factor for --audio_denoise.")
    parser.add_argument("--audio_denoise_floor", type=float, default=0.03,
                        help="Spectral-subtraction spectral floor (fraction of magnitude kept).")
    parser.add_argument("--lq_grayscale", action="store_true", default=False,
                        help="Convert LQ video to black & white before encoding, matching "
                             "training/validation (train degrades ~50%% to B&W). Use for old films.")
    parser.add_argument("--lq_gray_strength", type=float, default=1.0,
                        help="Desaturation strength when --lq_grayscale is set: 1.0=full gray, "
                             "0.9=heavy gray keeping 10%% of the LQ color. Only used with --lq_grayscale.")
    parser.add_argument("--stg_scale", type=float, default=0.0,
                        help="Spatio-temporal guidance scale (matches training val). 0=off. Try 1.0.")
    parser.add_argument("--stg_blocks", type=int, nargs="*", default=[29],
                        help="Transformer blocks to perturb for STG (val uses [29]).")
    parser.add_argument("--stg_mode", type=str, default="stg_av", choices=["stg_av", "stg_v"],
                        help="stg_av skips both video+audio self-attn; stg_v skips video only.")

    args = parser.parse_args()

    if not (args.prompt_cache and Path(args.prompt_cache).is_file()) and not args.text_encoder_path:
        parser.error("provide --text_encoder_path (or OMINIVR_TEXT_ENCODER_PATH) when no prompt cache is available")
    if args.tiny_decoder and not Path(args.tiny_decoder).is_file():
        parser.error(f"TinyDecoder checkpoint does not exist: {args.tiny_decoder}")

    # Discover input videos
    input_dir = Path(args.input_dir)
    video_paths = sorted(input_dir.glob("*.mp4"))
    if not video_paths:
        print(f"No .mp4 files found in {input_dir}")
        return
    print(f"Found {len(video_paths)} videos in {input_dir}")

    # Multi-GPU data parallel: fork workers
    if args.num_gpus > 1:
        import multiprocessing as mp
        strategies = [args.strategy] if args.strategy != "all" else ["no_cfg", "empty_cfg", "neg_cfg"]

        def _gpu_worker(gpu_id, assigned_paths, strategies, args):
            args_copy = argparse.Namespace(**vars(args))
            args_copy.gpu = gpu_id
            args_copy.num_gpus = 1  # each worker is single-GPU
            model_dict = load_model(args_copy)

            for strategy in strategies:
                out_dir = Path(args.output_dir) / strategy
                out_dir.mkdir(parents=True, exist_ok=True)

                for vpath in assigned_paths:
                    print(f"  [GPU {gpu_id}] {strategy}: {vpath.name}")
                    out_path = out_dir / vpath.name
                    if out_path.exists():
                        print(f"  [GPU {gpu_id}] {vpath.name} exists, skipping")
                        continue
                    try:
                        lq_video, fps, lq_audio, audio_sr = load_video(vpath, args.height, args.width, args.max_frames, args.resample_condition_frames, args.frame_rate)
                        if args.frame_rate > 0:
                            fps = float(args.frame_rate)
                        if not args.with_audio:
                            lq_audio = None
                        sr_video, sr_audio = restore_video(
                            model_dict, lq_video, lq_audio, audio_sr, fps,
                            strategy=strategy, guidance_scale=args.guidance_scale,
                            num_steps=args.num_steps, condition_noise=args.condition_noise, seed=args.seed,
                            target_frames=args.target_frames if args.target_frames > 0 else None,
                            noise_shift=args.noise_shift if args.noise_shift > 0 else None,
                            audio_denoise=args.audio_denoise,
                            audio_denoise_highpass=args.audio_denoise_highpass,
                            audio_denoise_over=args.audio_denoise_over,
                            audio_denoise_floor=args.audio_denoise_floor,
                            lq_grayscale=args.lq_grayscale,
                            lq_gray_strength=args.lq_gray_strength,
                            stg_scale=args.stg_scale,
                            stg_blocks=args.stg_blocks,
                            stg_mode=args.stg_mode,
                        )
                        tensor_debug_stats("sr_video_before_save", sr_video)
                        voc_sr = model_dict["vocoder"].output_sampling_rate if model_dict["vocoder"] is not None and sr_audio is not None else audio_sr
                        save_video(sr_video, out_path, fps, sr_audio, voc_sr)

                        if strategy == strategies[0]:
                            lq_dir = Path(args.output_dir) / "lq_input"
                            lq_dir.mkdir(parents=True, exist_ok=True)
                            lq_01 = (lq_video.squeeze(0).float() + 1) / 2
                            save_video(lq_01, lq_dir / vpath.name, fps,
                                       lq_audio.squeeze(0) if lq_audio is not None else None, audio_sr)

                        print(f"  [GPU {gpu_id}] {vpath.name} done")
                    except Exception as e:
                        print(f"  [GPU {gpu_id}] {vpath.name} ERROR: {e}")

        # Round-robin assign videos to GPUs
        assignments = [[] for _ in range(args.num_gpus)]
        for i, vp in enumerate(video_paths):
            assignments[i % args.num_gpus].append(vp)

        processes = []
        for gpu_id in range(args.num_gpus):
            if not assignments[gpu_id]:
                continue
            p = mp.Process(target=_gpu_worker, args=(gpu_id, assignments[gpu_id], strategies, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print(f"\nDone! Results in {args.output_dir}")
        return

    # Single-GPU path
    model_dict = load_model(args)

    strategies = [args.strategy] if args.strategy != "all" else ["no_cfg", "empty_cfg", "neg_cfg"]

    for strategy in strategies:
        out_dir = Path(args.output_dir) / strategy
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Strategy: {strategy} | guidance_scale: {args.guidance_scale if strategy != 'no_cfg' else 1.0}")
        print(f"Output: {out_dir}")
        print(f"{'='*60}")

        for i, vpath in enumerate(video_paths):
            print(f"\n[{i+1}/{len(video_paths)}] {vpath.name}")
            out_path = out_dir / vpath.name
            if out_path.exists():
                print(f"  Exists, skipping: {out_path}")
                continue

            try:
                lq_video, fps, lq_audio, audio_sr = load_video(vpath, args.height, args.width, args.max_frames, args.resample_condition_frames, args.frame_rate)
                if args.frame_rate > 0:
                    fps = float(args.frame_rate)
                print(f"  Loaded: {list(lq_video.shape)}, fps={fps:.1f}, audio={'yes' if lq_audio is not None else 'no'}")

                if not args.with_audio:
                    lq_audio = None

                sr_video, sr_audio = restore_video(
                    model_dict, lq_video, lq_audio, audio_sr, fps,
                    strategy=strategy,
                    guidance_scale=args.guidance_scale,
                    num_steps=args.num_steps,
                    condition_noise=args.condition_noise,
                    seed=args.seed,
                    target_frames=args.target_frames if args.target_frames > 0 else None,
                    noise_shift=args.noise_shift if args.noise_shift > 0 else None,
                    audio_denoise=args.audio_denoise,
                    audio_denoise_highpass=args.audio_denoise_highpass,
                    audio_denoise_over=args.audio_denoise_over,
                    audio_denoise_floor=args.audio_denoise_floor,
                    lq_grayscale=args.lq_grayscale,
                    lq_gray_strength=args.lq_gray_strength,
                    stg_scale=args.stg_scale,
                    stg_blocks=args.stg_blocks,
                    stg_mode=args.stg_mode,
                )

                tensor_debug_stats("sr_video_before_save", sr_video)
                voc_sr = model_dict["vocoder"].output_sampling_rate if model_dict["vocoder"] is not None and sr_audio is not None else audio_sr
                save_video(sr_video, out_path, fps, sr_audio, voc_sr)

                if strategy == strategies[0]:
                    lq_dir = Path(args.output_dir) / "lq_input"
                    lq_dir.mkdir(parents=True, exist_ok=True)
                    lq_01 = (lq_video.squeeze(0).float() + 1) / 2
                    save_video(lq_01, lq_dir / vpath.name, fps,
                               lq_audio.squeeze(0) if lq_audio is not None else None, audio_sr)

                print(f"  Saved: {out_path}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nDone! Results in {args.output_dir}")


if __name__ == "__main__":
    main()
