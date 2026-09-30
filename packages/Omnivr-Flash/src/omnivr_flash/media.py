"""Video-window loading and atomic AV export for Omnivr-Flash."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile
import wave

import av
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class InputWindow:
    video: torch.Tensor
    audio: torch.Tensor
    audio_rate: int
    width: int
    height: int
    frame_rate: float


def read_window(path: Path, frames: int, frame_rate: float, width: int, height: int, device: torch.device) -> InputWindow:
    """Read the first requested AV window; resize, resample, and edge-pad."""
    images = []
    with av.open(str(path)) as container:
        if not container.streams.video or not container.streams.audio:
            raise ValueError(f"Joint restoration requires both video and audio tracks: {path}")
        rate = container.streams.video[0].average_rate
        if rate is None or float(rate) <= 0:
            raise ValueError(f"Cannot determine the source video frame rate: {path}")
        source_rate = float(rate)
        target = 0
        for index, frame in enumerate(container.decode(video=0)):
            while target < frames and index >= round(target * source_rate / frame_rate):
                images.append(frame.reformat(width=width, height=height, format="rgb24").to_ndarray())
                target += 1
            if target >= frames:
                break
    if len(images) != frames:
        raise ValueError(
            f"Input has only {len(images)} frames at the requested playback rate; "
            f"requested {frames}. Use a smaller --frames value of the form 8n+1."
        )

    audio_rate = 48000
    target_samples = round(frames / frame_rate * audio_rate)
    chunks = []
    sample_count = 0
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=audio_rate)
        for frame in container.decode(audio=0):
            for chunk in resampler.resample(frame):
                samples = chunk.to_ndarray().copy()
                chunks.append(samples)
                sample_count += samples.shape[-1]
            if sample_count >= target_samples:
                break
        for chunk in resampler.resample(None):
            chunks.append(chunk.to_ndarray().copy())
    if not chunks:
        raise ValueError(f"The source audio track did not decode: {path}")
    waveform = np.concatenate(chunks, axis=-1)[..., :target_samples]
    if waveform.shape[-1] < target_samples:
        waveform = np.pad(waveform, ((0, 0), (0, target_samples - waveform.shape[-1])))

    video = torch.from_numpy(np.stack(images)).permute(3, 0, 1, 2).unsqueeze(0)
    video = video.to(device=device, dtype=torch.bfloat16).div_(127.5).sub_(1.0)
    pad_h, pad_w = (-height) % 32, (-width) % 32
    if pad_h or pad_w:
        video = F.pad(video, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
    audio = torch.from_numpy(waveform).unsqueeze(0).to(device=device, dtype=torch.float32)
    return InputWindow(video, audio, audio_rate, width, height, frame_rate)


def check_finite(video: torch.Tensor, waveform: torch.Tensor) -> None:
    for first in range(0, video.shape[2], 8):
        if not bool(torch.isfinite(video[:, :, first:first + 8]).all()):
            raise RuntimeError("Video model output contains NaN or Inf")
    if not bool(torch.isfinite(waveform).all()):
        raise RuntimeError("Audio model output contains NaN or Inf")


def inspect_output(path: Path, width: int, height: int, frames: int, frame_rate: float) -> dict:
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) != 1:
            raise RuntimeError("Export must contain one video track and one audio track")
        video, audio = container.streams.video[0], container.streams.audio[0]
        if (video.width, video.height, video.frames) != (width, height, frames):
            raise RuntimeError("Exported video dimensions or frame count do not match the request")
        if abs(float(video.average_rate) - frame_rate) > 1e-5:
            raise RuntimeError("Exported playback rate does not match the request")
        if audio.channels != 2:
            raise RuntimeError("Exported audio must be stereo")
        if video.duration is None or audio.duration is None:
            raise RuntimeError("Exported stream durations are unavailable")
        video_duration = float(video.duration * video.time_base)
        audio_duration = float(audio.duration * audio.time_base)
        if abs(video_duration - audio_duration) > 0.03:
            raise RuntimeError("Exported audio and video durations do not align")
        return {
            "width": width, "height": height, "frames": frames,
            "video_frame_rate": float(video.average_rate),
            "audio_sample_rate": audio.rate, "audio_channels": audio.channels,
        }


@torch.inference_mode()
def write_output(
    result, window: InputWindow, destination: Path, ffmpeg: str,
    *, overwrite: bool = False, verify: bool = False,
) -> dict:
    """Write every video frame and align restored audio to the same duration."""
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames = result.video.shape[2]
    if verify:
        check_finite(result.video, result.audio.waveform)
    with tempfile.TemporaryDirectory(prefix=".omnivr-flash-", dir=destination.parent) as directory:
        temp = Path(directory)
        video_path, audio_path = temp / "restored.mp4", temp / "audio.wav"
        waveform = result.audio.waveform.detach().float().cpu()
        if waveform.ndim == 3:
            waveform = waveform[0]
        if waveform.ndim != 2 or waveform.shape[0] != 2:
            raise RuntimeError("Vocoder output must contain two audio channels")
        target_samples = round(frames / window.frame_rate * result.audio.sampling_rate)
        waveform = waveform[..., :target_samples]
        if waveform.shape[-1] < target_samples:
            waveform = F.pad(waveform, (0, target_samples - waveform.shape[-1]))
        pcm = (waveform.clamp(-1, 1).t().numpy() * 32767).astype("<i2")
        with wave.open(str(audio_path), "wb") as writer:
            writer.setnchannels(2)
            writer.setsampwidth(2)
            writer.setframerate(result.audio.sampling_rate)
            writer.writeframes(pcm.tobytes())

        command = [
            ffmpeg, "-hide_banner", "-nostats", "-loglevel", "error", "-y", "-threads", "4",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{window.width}x{window.height}",
            "-r", str(window.frame_rate), "-i", "pipe:0", "-i", str(audio_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-threads", "4", "-movflags", "+faststart", str(video_path),
        ]
        with tempfile.TemporaryFile() as error_log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_log)
            try:
                for index in range(frames):
                    frame = result.video[0, :, index, :window.height, :window.width]
                    frame = frame.float().add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
                    rgb = frame.permute(1, 2, 0).contiguous().cpu().numpy()
                    process.stdin.write(rgb.tobytes())
                process.stdin.close()
                status = process.wait(timeout=300)
            except BaseException as error:
                process.kill()
                process.wait()
                if process.stdin is not None and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                error_log.seek(0)
                details = error_log.read(8192).decode(errors="replace")
                if isinstance(error, BrokenPipeError):
                    raise RuntimeError(f"FFmpeg stopped accepting frames: {details}") from error
                raise
            if status:
                error_log.seek(0)
                raise RuntimeError(f"FFmpeg export failed: {error_log.read(8192).decode(errors='replace')}")

        metadata = inspect_output(video_path, window.width, window.height, frames, window.frame_rate)
        if verify:
            check = subprocess.run(
                [ffmpeg, "-hide_banner", "-nostats", "-v", "error", "-i", str(video_path),
                 "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                capture_output=True, text=True, timeout=300,
            )
            if check.returncode or check.stderr.strip():
                raise RuntimeError(f"Export failed full AV decoding: {check.stderr}")
            metadata["full_av_decode_verified"] = True
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output was created by another process: {destination}")
        os.replace(video_path, destination)
    return metadata
