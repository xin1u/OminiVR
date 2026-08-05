# OminiVR

Joint audio-video restoration for degraded historical films. OminiVR is a
LoRA fine-tune (rank 384, ~3.8B trainable params) of Lightricks'
[LTX-2.3](https://github.com/Lightricks/LTX-2) 22B video generation model,
trained to jointly denoise/enhance both the picture and the soundtrack of
old, degraded film footage in a single pass.

Project page: https://xin1u.github.io/OminiVR_PAGE/

This repository contains **inference code**, **trained LoRA weights**, and
the **OmniBench evaluation set** used in the paper. Training code and
configs are not included in this release.

## How it works

OminiVR injects the low-quality (LQ) condition as extra input channels
rather than through a separate cross-attention/projection path: the
transformer's `patchify_proj` is expanded from `Linear(128, dim)` to
`Linear(256, dim)`, and at each denoising step the noisy latent is
channel-concatenated with the LQ latent before that projection. The LoRA
weights adapt the rest of the transformer to this new input distribution
while a `condition_noise` schedule keeps training/inference behavior close
to the base model's.

## Installation

OminiVR depends on Lightricks' `ltx-core` / `ltx-trainer` / `ltx-pipelines`
packages, which are not vendored here. Clone the
[LTX-2 repository](https://github.com/Lightricks/LTX-2) separately and point
`OMINIVR_DEPS_ROOT` at its `packages/` directory:

```bash
git clone https://github.com/xin1u/OminiVR.git
cd OminiVR
pip install -e .

git clone https://github.com/Lightricks/LTX-2.git /path/to/ltx-2
export OMINIVR_DEPS_ROOT=/path/to/ltx-2/packages
```

## Downloading weights and data

All large binaries (base model, LoRA checkpoints, evaluation set) are hosted
on Hugging Face rather than in this git repository. OminiVR's own weights,
data, and reference outputs are all in a single repo,
[`xin1u/OmniVR`](https://huggingface.co/xin1u/OmniVR):

| What | Source | Notes |
|---|---|---|
| Base model + Gemma text encoder | [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks) | official upstream weights; governed by the [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE) |
| OminiVR LoRA weights (`step_01800`, `step_02400`) | [`xin1u/OmniVR`](https://huggingface.co/xin1u/OmniVR)`/weights/` | see [NOTICE](NOTICE) — these are a Derivative of LTX-2 |
| TinyDecoder weights (`taeltx2_3_wide.pth`) | [`xin1u/OmniVR`](https://huggingface.co/xin1u/OmniVR)`/tinydecoder/` | see [`ckpt/tinydecoder/README.md`](ckpt/tinydecoder/README.md) |
| OmniBench evaluation set | [`xin1u/OmniVR`](https://huggingface.co/xin1u/OmniVR)`/omnibench/` | see [`data/omnibench/README.md`](data/omnibench/README.md) |
| Reference inference outputs | [`xin1u/OmniVR`](https://huggingface.co/xin1u/OmniVR)`/predictions/` | precomputed restorations for both tracks, for quick comparison without rerunning inference |

We release two LoRA checkpoints from the same training run
(`v5_tav2av_hq`), matching the two evaluation tracks in the paper:

- **`step_01800`** — used for the `no_gt` / real-footage (no-reference) track
- **`step_02400`** — used for the `with_gt` / talking-head (full-reference) track

```bash
huggingface-cli download xin1u/OmniVR weights/ominivr_lora_step_01800.safetensors --local-dir .
huggingface-cli download xin1u/OmniVR weights/ominivr_lora_step_02400.safetensors --local-dir .
huggingface-cli download xin1u/OmniVR tinydecoder/taeltx2_3_wide.pth --local-dir ckpt

export OMINIVR_MODEL_PATH=/path/to/ltx-2.3-22b-dev.safetensors
export OMINIVR_TEXT_ENCODER_PATH=/path/to/gemma-3-12b-it-qat-q4_0-unquantized
export OMINIVR_LORA_STRUCTURE=/path/to/ltx-2.3-22b-distilled-lora-384.safetensors
```

`OMINIVR_LORA_STRUCTURE` only needs to point at *some* checkpoint sharing the
LoRA's target-module/rank structure (used to reconstruct the PEFT config
before loading weights); it defaults to `--checkpoint` itself if unset.

## Inference

```bash
python scripts/infer.py \
    --input_dir data/omnibench/no_gt \
    --output_dir /path/to/output \
    --checkpoint /path/to/ominivr_lora_step_01800.safetensors \
    --strategy neg_cfg \
    --guidance_scale 3.0 \
    --num_steps 30 \
    --condition_noise 0.3 \
    --width 1920 --height 1088 --max_frames 121 --frame_rate 24 \
    --stg_scale 1.0 --stg_blocks 29 --stg_mode stg_av \
    --audio_denoise --audio_denoise_highpass 200 --audio_denoise_over 2.5 --audio_denoise_floor 0.03
```

Three CFG sampling strategies are available via `--strategy`:

- `no_cfg` — guidance_scale=1.0, single forward per step (fastest)
- `empty_cfg` — positive=SR prompt, negative=empty string (matches training distribution)
- `neg_cfg` — positive=SR prompt, negative=negative prompt (strongest enhancement, used above)
- `all` — runs all three strategies on the same inputs, for comparison

Pass `--lq_grayscale` when restoring genuinely black-and-white old film
footage — training degraded ~50% of clips to grayscale, so the model expects
a grayscale LQ signal for that case.

**Multi-GPU**: pass `--num_gpus N` to data-parallelize across `N` GPUs (each
GPU processes a disjoint subset of input videos; round-robin assignment).
There is no FSDP/single-video-sharded inference mode in this release.

**TinyDecoder**: if `ckpt/tinydecoder/taeltx2_3_wide.pth` is present,
inference uses the fast TinyDecoder path (keeps the transformer resident on
GPU). If absent, it falls back automatically to the full VAE decoder
(slower, tiled to control memory).

## Evaluation

See [`data/omnibench/README.md`](data/omnibench/README.md) for the dataset
layout and the `with_gt`/`no_gt` track split. The evaluation script itself
(PSNR/SSIM/LPIPS, PESQ/STOI/SI-SDR, LSE-C/D, AV-Align, NR-IQA, etc.) is not
included in this release — predictions and ground truth match by filename,
so any standard implementation of these metrics can be plugged in directly.

Precomputed reference outputs (`neg_cfg`, the strategy used for the paper's
numbers) for both tracks are available under `predictions/` in the
[`xin1u/OmniVR`](https://huggingface.co/xin1u/OmniVR) Hugging Face repo, if
you want to compare against them without rerunning inference yourself.

## License

OminiVR's own code (`src/`, `scripts/`, excluding the vendored TinyDecoder
noted below) is licensed under [Apache-2.0](LICENSE).

The released LoRA weights are a **Derivative of LTX-2** and are governed by
the [LTX-2 Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE),
not Apache-2.0 — see [NOTICE](NOTICE) for details. The base LTX-2.3 model and
Gemma text encoder must be obtained from their official sources and are
likewise subject to that license.

`src/ominivr/tiny_decoder.py` is adapted from Ollin Boer Bohan's
Seraena/TAESD (MIT License) — full attribution in [NOTICE](NOTICE).

## Citation

If you use OminiVR in your research, please cite:

```bibtex
@misc{ominivr2026,
  title  = {OminiVR: Joint Audio-Video Restoration for Degraded Historical Films},
  author = {xin1u},
  year   = {2026},
  url    = {https://xin1u.github.io/OminiVR_PAGE/}
}
```
