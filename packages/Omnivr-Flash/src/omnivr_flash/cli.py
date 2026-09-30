"""Command-line interface for Omnivr-Flash. Model imports are lazy for --help."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import shutil
import sys


DEFAULT_VAE_CONFIG = Path(__file__).resolve().parents[2] / "configs/turbo_vae.json"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Omnivr-Flash",
        description="Single-step joint audio/video restoration with fused LoRA and parallel Turbo VAE.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="A video file containing video and audio tracks")
    source.add_argument("--input-dir", "--input_dir", type=Path, help="Process one window from each video in this directory")
    parser.add_argument("--output", type=Path, help="Output MP4 for --input")
    parser.add_argument("--output-dir", "--output_dir", type=Path, help="Output directory for --input-dir")
    parser.add_argument("--model-path", "--model_path", type=Path, required=True, help="LTX-2.3 22B dev safetensors")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Additive-conditioning Omnivr-Flash LoRA safetensors")
    parser.add_argument("--turbo-vae", "--turbo_vae", type=Path, required=True, help="UltraTinyVAE EMA or student state dict")
    parser.add_argument("--prompt-cache", "--prompt_cache", type=Path, required=True, help="Fixed prompt embedding cache from Flash training")
    parser.add_argument("--vae-config", type=Path, default=DEFAULT_VAE_CONFIG, help="Turbo VAE architecture JSON")
    parser.add_argument("--width", type=int, default=1920, help="Export width; internal edge padding is automatic")
    parser.add_argument("--height", type=int, default=1080, help="Export height; internal edge padding is automatic")
    parser.add_argument("--frames", "--max_frames", type=int, default=121, help="Frames in the first output window; must equal 8n+1")
    parser.add_argument("--frame-rate", "--frame_rate", type=float, default=24.0, help="Output video playback rate")
    parser.add_argument("--device", default="cuda:0", help="CUDA device, relative to CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition-noise", type=float, default=0.3, help="LQ latent noise, matching the Flash configuration")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable with libx264 and AAC support")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    parser.add_argument("--verify-output", action="store_true", help="Check finite tensors and fully decode exported audio/video")
    parser.add_argument("--write-metadata", action="store_true", help="Save model settings and media properties beside each output")
    return parser


def prepare_jobs(args, parser) -> list[tuple[Path, Path]]:
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error("--width and --height must be positive even integers for yuv420p output")
    if args.frames < 1 or (args.frames - 1) % 8:
        parser.error("--frames must be a positive number of the form 8n+1")
    if not math.isfinite(args.frame_rate) or args.frame_rate <= 0:
        parser.error("--frame-rate must be finite and positive")
    if not math.isfinite(args.condition_noise) or args.condition_noise < 0:
        parser.error("--condition-noise must be finite and non-negative")
    for field in ["model_path", "checkpoint", "turbo_vae", "prompt_cache", "vae_config"]:
        if not getattr(args, field).is_file():
            parser.error(f"Missing {field}: {getattr(args, field)}")
    if args.input is not None:
        if not args.input.is_file() or args.output is None or args.output_dir is not None:
            parser.error("Use --input EXISTING_VIDEO --output RESULT.mp4")
        jobs = [(args.input.resolve(), args.output.resolve())]
    else:
        if not args.input_dir.is_dir() or args.output_dir is None or args.output is not None:
            parser.error("Use --input-dir EXISTING_DIRECTORY --output-dir RESULTS_DIRECTORY")
        files = sorted(path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
        if not files:
            parser.error("No supported videos found in --input-dir")
        jobs = [(path.resolve(), (args.output_dir / f"{path.stem}_restored.mp4").resolve()) for path in files]
    outputs = set()
    all_inputs = {source for source, _ in jobs}
    for source, destination in jobs:
        if destination.suffix.lower() != ".mp4":
            parser.error("Output files must use the .mp4 extension")
        if destination in all_inputs:
            parser.error(f"Output would replace an input video: {destination}")
        if destination in outputs:
            parser.error(f"Input names would produce the same output: {destination}")
        outputs.add(destination)
        targets = [destination]
        if args.write_metadata:
            targets.append(Path(str(destination) + ".json"))
        for target in targets:
            if target.exists() and not args.overwrite:
                parser.error(f"Output exists; choose another path or use --overwrite: {target}")
    executable = shutil.which(args.ffmpeg)
    if executable is None:
        parser.error("FFmpeg was not found; install it or pass --ffmpeg /path/to/ffmpeg")
    args.ffmpeg = executable
    return jobs


def run(args, jobs) -> None:
    from .loading import ModelPaths
    from .media import read_window, write_output
    from .pipeline import FlashRestorer

    paths = ModelPaths(
        model=args.model_path, checkpoint=args.checkpoint, turbo_vae=args.turbo_vae,
        prompt_cache=args.prompt_cache, vae_config=args.vae_config,
    )
    restorer = FlashRestorer(paths, device=args.device, seed=args.seed, condition_noise=args.condition_noise)
    logger = logging.getLogger("Omnivr-Flash")
    for source, destination in jobs:
        logger.info("Restoring %s", source.name)
        window = read_window(source, args.frames, args.frame_rate, args.width, args.height, restorer.device)
        result = restorer.restore(window.video, window.audio, window.audio_rate, window.frame_rate)
        media = write_output(result, window, destination, args.ffmpeg, overwrite=args.overwrite, verify=args.verify_output)
        if args.write_metadata:
            metadata = {
                "project": "Omnivr-Flash", "input": str(source), "output": str(destination),
                "model": str(paths.model.resolve()), "checkpoint": str(paths.checkpoint.resolve()),
                "turbo_vae": str(paths.turbo_vae.resolve()), "prompt_cache": str(paths.prompt_cache.resolve()),
                "vae_config": str(paths.vae_config.resolve()), "dtype": "bfloat16", "batch_size": 1,
                "denoising_steps": 1, "sigmas": [1.0, 0.0], "lora_merged": True,
                "vae_parallel": True, "seed": args.seed, "condition_noise": args.condition_noise,
                "model_width": window.video.shape[-1], "model_height": window.video.shape[-2],
                "video_latent_shape": result.video_latent_shape, "audio_latent_shape": result.audio_latent_shape,
                "media": media,
            }
            sidecar = Path(str(destination) + ".json")
            sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        logger.info("Saved %s", destination)
        del result, window


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    jobs = prepare_jobs(args, parser)
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    try:
        run(args, jobs)
    except (RuntimeError, ValueError, OSError) as error:
        logging.getLogger("Omnivr-Flash").error("%s", error)
        sys.exit(1)
