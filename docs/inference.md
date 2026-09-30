# OmniVR and OmniVR-Flash inference guide

Single-step joint audio/video restoration built on LTX-2.3, a fused Flash LoRA,
and an UltraTinyVAE video encoder and decoder.

The original OmniVR multi-step inference path is also included. Use the root
`infer.py` for Omnivr-Flash, or
`packages/omini-restore-trainer/scripts/infer.py` for the original multi-step
sampler with CFG/STG and channel-concat conditioning.

The inference path keeps the models on one CUDA device and uses BF16,
PyTorch scaled-dot-product attention, cached text embeddings, and the native
parallel Turbo VAE. Video and audio are restored together with one Euler update
at `sigma = [1, 0]`. The audio decoder and vocoder generate the output waveform.

## Layout

```text
OminiVR/
├── infer.py                         # Omnivr-Flash entry point
├── requirements.txt
├── LICENSE
└── packages/
    ├── Omnivr-Flash/
    │   ├── scripts/infer.py         # Direct package entry point
    │   ├── configs/turbo_vae.json   # Architecture matching the Turbo VAE weights
    │   └── src/omnivr_flash/
    │       ├── cli.py
    │       ├── model_loader.py     # Inference-only LTX component loaders
    │       ├── loading.py          # Strict checkpoint loading and LoRA fusion
    │       ├── pipeline.py         # One-step joint AV restoration
    │       ├── media.py            # Input preparation and AV export
    │       └── turbo_vae.py        # Bundled UltraTinyVAE implementation
    ├── ltx-core/                    # Shared LTX model definitions and runtime
    ├── ltx-pipelines/               # Original LTX inference pipelines
    └── omini-restore-trainer/       # Original path retained for existing commands
        └── scripts/
            ├── infer.py           # Original multi-step OmniVR entry point
            ├── inference_model_loader.py # Model and encoder loaders
            ├── inference_gemma_8bit.py # Optional 8-bit text encoder loader
            └── inference_support.py # Channel-concat and checkpoint helpers
```

Both `python infer.py --help` and
`python packages/Omnivr-Flash/scripts/infer.py --help` describe the Flash CLI.
The launchers resolve the bundled source directories relative to their own
location. There is no dependency on a cluster checkout or training repository.
Both inference paths include their model-loading and conditioning helpers.
Training loops, training launchers, datasets, losses, and optimizer setup are
excluded. The original helper file named `train.py` is now
`inference_support.py`; the multi-step entry point imports it directly.

## Environment

Use Python 3.12, a CUDA GPU with BF16 support, and FFmpeg with `libx264` and AAC
encoding. The following installs the CUDA 12.8 PyTorch packages:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1
python -m pip install -r requirements.txt
ffmpeg -version
```

The bundled source packages are loaded directly by the launchers; editable
installation is not required. TorchCodec is included for the original
multi-step entry point's Torchaudio audio export.

## Required weights

Supply the following files explicitly. Model weights and input media are not
included in this code package.

| Argument | Required file |
|---|---|
| `--model-path` | LTX-2.3 22B dev `.safetensors` checkpoint |
| `--checkpoint` | Trained **Omnivr-Flash** LoRA with additive video/audio condition projections |
| `--turbo-vae` | UltraTinyVAE EMA or student state dict, `.pth` or `.safetensors` |
| `--prompt-cache` | Fixed prompt embedding cache with video and audio contexts |
| `--vae-config` | Optional architecture JSON; defaults to the bundled `configs/turbo_vae.json` |

The base model is available from
[Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3).
The other files come from the corresponding Flash and Turbo VAE training runs.

The Flash checkpoint must contain paired
`diffusion_model.<module>.lora_A.weight` /
`diffusion_model.<module>.lora_B.weight` tensors and all four additive condition
tensors (`_cond_video_proj` and `_cond_audio_proj`, weight and bias). Its saved
B tensors already include LoRA scaling. They are fused once as `W + B @ A` in
FP32, then cast to BF16. The loader rejects unmatched tensors and incompatible
channel-concatenation checkpoints.

The prompt cache uses the existing training format:

```python
{
    "val_positive": {
        "video_encoding": tensor,  # [1, text_tokens, 4096]
        "audio_encoding": tensor,  # [1, text_tokens, 2048]
    },
    # Other training-cache fields may be present.
}
```

Turbo VAE weights are loaded strictly. The bundled architecture uses 128 latent
channels, total spatial compression of 32, and temporal compression of 8.
The decoder's initial seven causal warmup frames are trimmed in the same way
as in its training implementation.

## Restore a clip window

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --input /path/to/input.mp4 \
    --output /path/to/restored.mp4 \
    --model-path /path/to/ltx-2.3-22b-dev.safetensors \
    --checkpoint /path/to/flash_lora.safetensors \
    --turbo-vae /path/to/ema_student.pth \
    --prompt-cache /path/to/sr_prompt_embeddings.pt \
    --width 1920 --height 1080 \
    --frames 121 --frame-rate 24
```

`--frames` selects the **first window** after resampling the input. It must
have the form `8n+1`. This command processes one window, even if the source
video is longer. For a shorter input, choose a smaller valid value; insufficient
video frames cause an explicit error. Both video and audio tracks are required.
`--frame-rate` specifies the output video's playback rate.

Input video is resized to the requested width and height, then edge-padded to
multiples of 32 for the model. The output is cropped back to the requested
size. Width and height must be even for the exported `yuv420p` video.

Examples of output dimensions:

| Preset | Width | Height |
|---|---:|---:|
| 480p | 854 | 480 |
| 720p | 1280 | 720 |
| 1080p | 1920 | 1080 |
| 1440p | 2560 | 1440 |

Audio is resampled to stereo for conditioning. Restored audio is trimmed or
zero-padded to the video duration before muxing, so a short audio track cannot
truncate the video. The output is H.264 video plus AAC stereo audio in MP4.

## Process a directory

Models are loaded once and reused for all files. Each source produces one
window and is written as `<source_stem>_restored.mp4`:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --input-dir /path/to/inputs \
    --output-dir /path/to/restored \
    --model-path /path/to/ltx-2.3-22b-dev.safetensors \
    --checkpoint /path/to/flash_lora.safetensors \
    --turbo-vae /path/to/ema_student.pth \
    --prompt-cache /path/to/sr_prompt_embeddings.pt \
    --width 1280 --height 720 --frames 121 --frame-rate 24
```

`--device` is relative to `CUDA_VISIBLE_DEVICES`. The default is `cuda:0`.
Existing output files are protected unless `--overwrite` is supplied. Input
files cannot be selected as output destinations.

## Output checks and metadata

- `--verify-output` checks that model outputs are finite and fully decodes the
  generated audio/video with FFmpeg. It does not perform perceptual assessment.
- `--write-metadata` creates `<output.mp4>.json` with model paths, inference
  settings, latent shapes, and output media properties.
- `--ffmpeg /path/to/ffmpeg` selects a specific FFmpeg executable.
- `--condition-noise` defaults to `0.3`, matching the Flash configuration.
- `--seed` defaults to `42`.

Spatial tiling, CPU offload, quantization, and extra guidance passes are not
part of this inference path. The parallel model execution requires sufficient
memory for the requested window.

## Original multi-step OmniVR inference

The original entry point retains the Euler sampler with a configurable step
count, `no_cfg` / `empty_cfg` / `neg_cfg` strategies, STG, tiled VAE decoding,
audio postprocessing, and per-video distribution across multiple GPUs.

```bash
CUDA_VISIBLE_DEVICES=0 python packages/omini-restore-trainer/scripts/infer.py \
    --input_dir /path/to/lq_videos \
    --output_dir /path/to/restored \
    --model_path /path/to/ltx-2.3-22b-dev.safetensors \
    --text_encoder_path /path/to/gemma \
    --checkpoint /path/to/omnivr_concat_lora.safetensors \
    --prompt_cache /path/to/v5_sr_prompt_embeddings.pt \
    --strategy neg_cfg --guidance_scale 3.0 --num_steps 30 \
    --width 1920 --height 1088 --max_frames 121 --frame_rate 24
```

Use `--strategy all` to run all three CFG strategies. Use `--num_gpus N`
with N visible GPUs for the original data-parallel mode. STG options include
`--stg_scale`, `--stg_blocks`, and `--stg_mode`; audio postprocessing is
enabled with `--audio_denoise`. Run the original entry point with `--help`
for all options.

This path uses the original channel-concat OmniVR LoRA and the full LTX video
VAE. Its checkpoint is different from the additive-conditioning Flash LoRA
used by the root entry point. The cached prompt format includes
`val_positive`, `val_negative`, and `empty_prompt`; a missing cache selects
live Gemma encoding. The legacy directory name is retained so existing
entry-point paths continue to work.

The model loaders, optional 8-bit Gemma helper, channel-concat conditioning,
and LoRA rank discovery are bundled in the `inference_*.py` modules beside
the multi-step entry point. They load independently of the training package
and the additional upstream LTX pipelines.

## Release integration

The repository root also provides `infer_multistep.py`; the old
`scripts/infer.py` command forwards to it. The full LTX video VAE is the
multistep default. `--tiny_decoder` explicitly selects the bundled legacy
TAEHV helper and its separate decoder-only checkpoint. Gemma is required
only when no compatible prompt cache is available.

See the root README for current public weight availability and the distinction
between this window-based export and the optimized pipeline in the paper.

## Sources and license

The bundled LTX components retain the LTX-2 Community License Agreement in
`LICENSE`. The Turbo VAE implementation comes from the project-provided
`UltraForcing/xvideo/models/ultra_latent_up/ultra_tiny_vae.py`; its architecture
is specified by the bundled JSON. Omnivr-Flash adds the checkpoint loader,
joint one-step restoration path, and media/CLI integration.
