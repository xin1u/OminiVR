# OmniBench — OminiVR Joint Audio-Video Restoration Benchmark

200 clips, 5.04s / 24fps / 121 frames, split into two groups — **evaluation
protocol depends on whether ground truth exists**.

```
omnibench/
├── README.md          this file
├── manifest.json       200-entry metadata (group / has_gt / resolution / bw / speech_ratio / source)
│
├── with_gt/            71 "talking-head" clips — has GT, full-reference (controlled) eval
│   ├── lq/             degraded input (fed to the model; paired by filename with gt/)
│   └── gt/             clean ground truth (frame-aligned with lq/, same filename)
│
└── no_gt/              129 real old-film clips — no GT, reference-free (real) eval only
                        (the old film itself is the input; no clean reference exists)
```

## Download

This directory ships without the actual video files — they are hosted in the
[`xin1u/OmniVR`](https://huggingface.co/xin1u/OmniVR) Hugging Face repo,
under `omnibench/`:

```bash
huggingface-cli download xin1u/OmniVR --include "omnibench/*" --local-dir data
# manifest.json, with_gt/, no_gt/ land under data/omnibench/, matching the layout below
```

## What the two groups are for

| | `with_gt/` (71) | `no_gt/` (129) |
|---|---|---|
| Content | talking-head / speech | real old film footage |
| Source | LQ synthesized from HQ source via the degradation pipeline; GT = clean HQ | collected real old films (already degraded) |
| GT | ✅ `gt/`, frame-aligned, same filename | ❌ none |
| Eval track | **controlled** (full-reference) | **real** (no-reference) |
| Applicable metrics | video PSNR/SSIM/LPIPS; audio PESQ/STOI/SI-SDR; AV sync LSE-C/D, AV-Align; NR-IQA also applicable | NR-IQA only: MUSIQ/CLIP-IQA/MANIQA + no-reference audio metrics |
| B&W / color | 57 B&W / 14 color (GT color) | old-film style (mostly B&W) |

`with_gt` is purpose-built for lip-sync (LSE-C/D) and audio-video sync
ablations: every clip was verified via face detection (speaker on-screen) and
silero-VAD (speech ratio ≥0.40).

## Usage

**Inference** (run restoration on each group's inputs):
```bash
# with_gt group (has GT, controlled eval)
python scripts/infer.py --input_dir data/omnibench/with_gt/lq  --output_dir <out_wg>  ...
# no_gt group (real old films)
python scripts/infer.py --input_dir data/omnibench/no_gt       --output_dir <out_ng>  ...
```

**Evaluation**: this repository does not include an evaluation script (out of
scope for this release — see the main README). Predictions and ground truth
are matched **by filename** (`with_gt/lq/X.mp4` ↔ `with_gt/gt/X.mp4`); for the
`controlled` track compute full-reference video/audio/AV-sync metrics against
`with_gt/gt/`, and for the `real` track compute no-reference metrics directly
on the `no_gt/` predictions.

## Data spec

- Resolution: original, not downsampled. `with_gt` is mostly 1920×1080, with a
  handful of 1440p/4K clips; `no_gt` keeps each old film's original resolution.
- Frames/rate: 121 frames @ 24fps = 5.04s.
- Audio: 48kHz. `with_gt/lq` has legacy old-film audio degradation applied
  (hiss/rumble/band-limiting).
- Per-clip metadata: see `manifest.json`.
