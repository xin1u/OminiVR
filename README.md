<div align="center">

# OmniVR

### Joint Audio-Video Conditional Generation for Archival Footage Restoration

**Xin Lu**<sup>†</sup> · **Zihao Fan**<sup>†</sup> · **Jie Huang**<sup>‡</sup> · **Mingchen Zhong** · **Hexin Zhang** · **Xueyang Fu**<sup>✉</sup> · **Zheng-Jun Zha**

University of Science and Technology of China

<sup>†</sup> Equal contribution · <sup>‡</sup> Project leader · <sup>✉</sup> Corresponding author

[Latest manuscript · 30 Sep 2026](https://xin1u.github.io/OminiVR_PAGE/assets/OmniVR.pdf) · [arXiv:2608.04224](https://arxiv.org/abs/2608.04224) · [Project page](https://xin1u.github.io/OminiVR_PAGE/) · [GitHub](https://github.com/xin1u/OminiVR) · [Hugging Face](https://huggingface.co/xin1u/OmniVR)

</div>

**30 September 2026:** updated manuscript metadata and framework; complete Flash and multistep inference source is included below. Public multistep weights, benchmark files, and existing reference outputs remain available.

## Abstract

Archival footage often suffers from coupled visual and acoustic degradations, yet most restoration systems process the two modalities separately. To address this problem, we present OmniVR, the first systematic framework for joint audio-video restoration, covering data construction, model adaptation, efficient inference, and evaluation. We construct a high-quality audio-video corpus with detailed captions and use a joint degradation pipeline to produce aligned clean and degraded pairs. Using these pairs, we adapt a pretrained text-to-audio-video model (T2AV) by introducing degraded audio-video conditions (TAV2AV), then progressively replace sample captions with a fixed restoration prompt while retaining caption/null rehearsal. The resulting AV2AV model requires no user-provided text. Under a compatible residual-learning model, we prove that this condition-annealing schedule reduces gradient variance and expected restoration risk relative to direct fixed-prompt adaptation at the same training budget. For efficient deployment, OmniVR-Flash combines reduced-resolution video conditioning, MeanFlow-based one-step distillation, and Turbo VAE, achieving approximately 38 fps at 1K and 18 fps at 2K on a single B200 GPU. We further introduce OmniVRBench to evaluate four complementary dimensions: visual quality, audio quality, temporal consistency, and audio-visual synchrony. OmniVR achieves state-of-the-art results on public benchmarks and OmniVRBench. Data, code, and model weights will be released.

## Framework

![OmniVR and OmniVR-Flash framework](assets/architecture.png)

OmniVR and OmniVR-Flash framework. (a) Conditional SFT adapts a joint audiovisual DiT with degraded inputs and caption annealing to a fixed prompt. (b) AV MeanFlow enables one-step prediction. (c) Turbo VAE accelerates video encoding and decoding. (d) OmniVR-Flash downsamples LQ video before condition encoding and DiT injection, reducing conditioning cost while preserving high-resolution outputs. The last restored frame anchors the next window.

[Vector framework (PDF)](assets/architecture.pdf) · [Latest manuscript](https://xin1u.github.io/OminiVR_PAGE/assets/OmniVR.pdf)

## Install

Use Python 3.12, a CUDA GPU with BF16 support, and FFmpeg with H.264/AAC
encoding. The source packages are bundled and loaded relative to the
entry-point files; no external LTX checkout or training package is needed.

```bash
git clone https://github.com/xin1u/OminiVR.git
cd OminiVR
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1
python -m pip install -r requirements.txt
ffmpeg -version
python infer.py --help
python infer_multistep.py --help
```

The 22B backbone and video window must fit the selected device. Flash keeps
the models resident on one GPU; multistep inference supports VAE tiling and
model offload. Neither command shards one video across multiple GPUs.

## Inference

| | OmniVR-Flash | OmniVR multistep |
|---|---|---|
| Entry point | `infer.py` | `infer_multistep.py` |
| Sampling | One Euler update, `sigma = [1, 0]` | Configurable Euler steps, CFG and STG |
| Video VAE | UltraTinyVAE encoder and decoder | Full LTX video VAE by default |
| Restoration checkpoint | Additive-conditioning Flash LoRA | Channel-concat OmniVR LoRA |
| Text | Fixed prompt cache | Fixed prompt cache or live Gemma encoding |
| Input | File or directory; video and audio required | Directory of MP4 files |

### OmniVR-Flash

Flash requires **three compatible project artifacts**: its additive-conditioning
LoRA, an UltraTinyVAE checkpoint matching the bundled architecture JSON, and
the fixed prompt cache. These artifacts are **not yet included in the public
model repository**. The existing `ominivr_lora_step_01800/02400` and
`taeltx2_3_wide.pth` files are for the multistep/legacy decoder path and cannot
be substituted for Flash weights.

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --input /path/to/input.mp4 \
    --output /path/to/restored.mp4 \
    --model-path /path/to/ltx-2.3-22b-dev.safetensors \
    --checkpoint /path/to/flash_lora.safetensors \
    --turbo-vae /path/to/turbo_vae.pth \
    --prompt-cache /path/to/flash_prompt_embeddings.pt \
    --width 1920 --height 1080 --frames 121 --frame-rate 24 \
    --verify-output --write-metadata
```

For directory input, replace `--input` / `--output` with `--input-dir` /
`--output-dir`. The models are loaded once and reused. Each source produces
its first requested window. `--frames` must be `8n+1`; short inputs produce
an error. Dimensions are padded to multiples of 32 internally and cropped
back for export. Existing outputs require `--overwrite` to replace them.

### OmniVR multistep

Obtain the base model from [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3)
and the text encoder from
[google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized),
following their access and license terms. Download a public OmniVR checkpoint:

```bash
hf download xin1u/OmniVR weights/ominivr_lora_step_01800.safetensors --local-dir .
# Controlled-track alternative:
hf download xin1u/OmniVR weights/ominivr_lora_step_02400.safetensors --local-dir .

CUDA_VISIBLE_DEVICES=0 python infer_multistep.py \
    --input_dir /path/to/lq_videos \
    --output_dir /path/to/restored \
    --model_path /path/to/ltx-2.3-22b-dev.safetensors \
    --text_encoder_path /path/to/gemma-3-12b-it-qat-q4_0-unquantized \
    --checkpoint weights/ominivr_lora_step_01800.safetensors \
    --strategy neg_cfg --guidance_scale 3.0 --num_steps 30 \
    --condition_noise 0.3 \
    --width 1920 --height 1088 --max_frames 121 --frame_rate 24
```

The public `step_01800` checkpoint is associated with the real/no-reference
track; `step_02400` is associated with the controlled/full-reference track.
The example uses the existing 30-step command; the paper's Flash efficiency
comparison uses a 15-step base model. Sampling and guidance settings affect
both output and runtime.

`--strategy` also accepts `no_cfg`, `empty_cfg`, and `all`. Optional STG is
controlled by `--stg_scale`, `--stg_blocks`, and `--stg_mode`. Use
`--num_gpus N` to distribute different input videos across N visible GPUs.
`--audio_denoise` enables the optional audio postprocessing path.

The old command `python scripts/infer.py ...` remains available. Existing
`OMINIVR_MODEL_PATH`, `OMINIVR_TEXT_ENCODER_PATH`, and
`OMINIVR_LORA_STRUCTURE` environment variables are accepted. A compatible
`--prompt_cache` skips Gemma loading; otherwise Gemma is required. For the
optional decoder-only TAEHV checkpoint, see
[the legacy TinyDecoder instructions](ckpt/tinydecoder/README.md).

### Release scope

Both commands include their model loaders, conditioning code, media I/O,
and shared LTX runtime. The supplied export processes clip windows. It does
not expose the reduced-resolution Flash conditioning or automatic
last-frame window chaining depicted in the paper, and its runtime has not
been equated with the paper's optimized benchmark. Training loops and
evaluation scripts are outside this inference release.

See [the detailed inference guide](docs/inference.md) for checkpoint tensor
formats, prompt-cache fields, VAE geometry, tiling, and output validation.

## OmniVRBench and public artifacts

OmniVRBench contains **200 clips**: 71 controlled clips with clean targets
and 129 real archival clips without ground truth. Evaluation covers visual
quality, audio quality, temporal consistency, and audiovisual synchrony.

| Artifact | Location |
|---|---|
| Multistep LoRA checkpoints | [https://huggingface.co/xin1u/OmniVR/tree/main/weights](https://huggingface.co/xin1u/OmniVR/tree/main/weights) |
| Optional legacy TinyDecoder | [https://huggingface.co/xin1u/OmniVR/tree/main/tinydecoder](https://huggingface.co/xin1u/OmniVR/tree/main/tinydecoder) |
| Benchmark and manifest | [https://huggingface.co/xin1u/OmniVR/tree/main/omnibench](https://huggingface.co/xin1u/OmniVR/tree/main/omnibench) |
| Existing multistep reference outputs | [https://huggingface.co/xin1u/OmniVR/tree/main/predictions](https://huggingface.co/xin1u/OmniVR/tree/main/predictions) |

## Paper-reported efficiency

The September 30 manuscript reports the following measurements on one
NVIDIA B200 for 121-frame windows (Tables 22–23):

| Configuration | Steps | 1920 × 1088 (1K) | 2560 × 1440 (2K) |
|---|---:|---:|---:|
| OmniVR base | 15 | 3.5 fps | 1.5 fps |
| OmniVR-Flash, all three optimizations | 1 | 38.2 fps | 17.8 fps |

The Flash configuration in the paper combines reduced-resolution
conditioning, MeanFlow one-step distillation, and Turbo VAE. These are
paper measurements, not new benchmarks of this exported code.

## License and acknowledgments

The current inference bundle and LTX-derived model weights use the
[LTX-2 Community License Agreement](LICENSE), including its attachments.
Bundled LTX components retain their upstream notices. The optional legacy
TAEHV implementation retains its MIT attribution in [NOTICE](NOTICE).
Earlier Apache-2.0 attribution is retained in `licenses/Apache-2.0.txt`.
The Google Gemma text encoder remains subject to its own upstream terms.

We acknowledge [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) and
[Seraena/TAESD](https://github.com/madebyollin/seraena).

## Citation

```bibtex
@article{lu2026omnivr,
  title={OmniVR: Joint Audio-Video Conditional Generation for Archival Footage Restoration},
  author={Lu, Xin and Fan, Zihao and Huang, Jie and Zhong, Mingchen and Zhang, Hexin and Fu, Xueyang and Zha, Zheng-Jun},
  journal={arXiv preprint arXiv:2608.04224},
  year={2026},
  eprint={2608.04224},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.04224}
}
```

Contact: [luxion@mail.ustc.edu.cn](mailto:luxion@mail.ustc.edu.cn).
