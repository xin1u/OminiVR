"""Single-step, batch-one Omnivr-Flash inference with parallel Turbo VAE."""

from __future__ import annotations

from dataclasses import dataclass, replace
import gc

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.model.audio_vae import AudioProcessor, decode_audio, encode_audio
from ltx_core.model.transformer.attention import Attention, PytorchAttention
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import X0Model
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import Audio, AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape

from . import logger
from .loading import ModelPaths, PromptEmbeddings, load_prompt_cache, load_turbo_vae, merge_flash_checkpoint
from .model_loader import load_audio_vae_decoder, load_audio_vae_encoder, load_transformer, load_vocoder


@dataclass
class Restoration:
    video: torch.Tensor
    audio: Audio
    video_latent_shape: tuple[int, ...]
    audio_latent_shape: tuple[int, ...]


@torch.inference_mode()
def denoise_one_step(
    transformer: torch.nn.Module,
    lq_video: torch.Tensor,
    lq_audio: torch.Tensor,
    prompts: PromptEmbeddings,
    frame_rate: float,
    seed: int,
    condition_noise: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The trained sigma=[1, 0] joint AV Euler path, with additive LQ input."""
    device = lq_video.device
    _, _, latent_frames, latent_height, latent_width = lq_video.shape
    scale = SpatioTemporalScaleFactors.default()
    video_patchifier = VideoLatentPatchifier(patch_size=1)
    audio_patchifier = AudioPatchifier(patch_size=1)
    pixel_shape = VideoPixelShape(
        batch=1, frames=(latent_frames - 1) * scale.time + 1,
        height=latent_height * scale.height, width=latent_width * scale.width,
        fps=frame_rate,
    )
    video_tools = VideoLatentTools(
        patchifier=video_patchifier, target_shape=VideoLatentShape.from_pixel_shape(pixel_shape),
        fps=frame_rate, scale_factors=scale, causal_fix=True,
    )
    noiser = GaussianNoiser(torch.Generator(device=device).manual_seed(seed))
    video_state = noiser(video_tools.create_initial_state(device, torch.bfloat16), noise_scale=1.0)
    audio_tools = AudioLatentTools(
        patchifier=audio_patchifier, target_shape=AudioLatentShape.from_torch_shape(lq_audio.shape),
    )
    audio_state = noiser(audio_tools.create_initial_state(device, torch.bfloat16), noise_scale=1.0)
    audio_condition = audio_patchifier.patchify(lq_audio)
    video_condition = video_patchifier.patchify(lq_video)
    if condition_noise > 0:
        video_condition = video_condition + torch.randn_like(video_condition) * condition_noise
        audio_condition = audio_condition + torch.randn_like(audio_condition) * condition_noise
    if video_condition.shape != video_state.latent.shape:
        raise ValueError("Video condition and target token shapes must match")

    sigmas = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)
    sigma = sigmas[0]
    stepper = EulerDiffusionStep()
    x0_model = X0Model(transformer)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        video_mod = Modality(
            enabled=True, latent=video_state.latent, sigma=sigma.repeat(1),
            timesteps=sigma * video_state.denoise_mask, positions=video_state.positions,
            context=prompts.video_context_positive, context_mask=None, cond_latent=video_condition,
        )
        audio_mod = Modality(
            enabled=True, latent=audio_state.latent, sigma=sigma.repeat(1),
            timesteps=sigma * audio_state.denoise_mask, positions=audio_state.positions,
            context=prompts.audio_context_positive, context_mask=None, cond_latent=audio_condition,
        )
        denoised_video, denoised_audio = x0_model(video=video_mod, audio=audio_mod, perturbations=None)
        if denoised_audio is None:
            raise RuntimeError("The joint model did not return an audio prediction")
        video_state = replace(video_state, latent=stepper.step(
            sample=video_mod.latent, denoised_sample=denoised_video, sigmas=sigmas, step_index=0,
        ))
        audio_state = replace(audio_state, latent=stepper.step(
            sample=audio_mod.latent, denoised_sample=denoised_audio, sigmas=sigmas, step_index=0,
        ))
    video_state = video_tools.unpatchify(video_tools.clear_conditioning(video_state))
    audio_state = audio_tools.unpatchify(audio_tools.clear_conditioning(audio_state))
    return video_state.latent, audio_state.latent


class FlashRestorer:
    """Load once and reuse the models for successive clip windows on one GPU."""

    def __init__(self, paths: ModelPaths, device: str = "cuda:0", seed: int = 42, condition_noise: float = 0.3):
        paths.validate()
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Omnivr-Flash requires a CUDA GPU")
        torch.cuda.set_device(self.device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU must support BF16")
        if condition_noise < 0:
            raise ValueError("condition_noise must be non-negative")
        self.paths = paths
        self.seed = seed
        self.condition_noise = condition_noise
        torch.set_num_threads(8)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.manual_seed(seed)
        logger.info("Loading Omnivr-Flash models")
        self.transformer = load_transformer(paths.model, device=self.device, dtype=torch.bfloat16)
        self.transformer.init_cond_proj()
        self.transformer.to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        for module in self.transformer.modules():
            if isinstance(module, Attention):
                module.attention_function = PytorchAttention()
        self.checkpoint_info = merge_flash_checkpoint(self.transformer, paths.checkpoint)
        self.prompts = load_prompt_cache(paths.prompt_cache, self.device)
        self.audio_encoder = load_audio_vae_encoder(paths.model, self.device, torch.bfloat16).eval().requires_grad_(False)
        self.audio_decoder = load_audio_vae_decoder(paths.model, self.device, torch.bfloat16).eval().requires_grad_(False)
        self.vocoder = load_vocoder(paths.model, self.device, torch.bfloat16).eval().requires_grad_(False)
        self.audio_processor = AudioProcessor(
            target_sample_rate=self.audio_encoder.sample_rate,
            mel_bins=self.audio_encoder.mel_bins, mel_hop_length=self.audio_encoder.mel_hop_length,
            n_fft=self.audio_encoder.n_fft,
        ).to(self.device)
        self.turbo_vae, self.trim_frames = load_turbo_vae(paths, self.device)
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Omnivr-Flash models ready")

    @torch.inference_mode()
    def restore(self, video: torch.Tensor, audio: torch.Tensor, audio_rate: int, frame_rate: float) -> Restoration:
        if video.ndim != 5 or video.shape[:2] != (1, 3):
            raise ValueError("Video must be [1, 3, frames, height, width] in [-1, 1]")
        if (video.shape[2] - 1) % 8 or video.shape[-2] % 32 or video.shape[-1] % 32:
            raise ValueError("Video needs 8n+1 frames and spatial dimensions divisible by 32")
        if audio.ndim != 3 or audio.shape[:2] != (1, 2) or audio.shape[-1] == 0:
            raise ValueError("Audio must be a nonempty [1, 2, samples] waveform")
        if audio_rate <= 0 or frame_rate <= 0:
            raise ValueError("Audio sampling rate and video frame rate must be positive")
        video = video.to(device=self.device, dtype=torch.bfloat16)
        audio = audio.to(device=self.device, dtype=torch.float32)
        torch.manual_seed(self.seed)
        video_latent = self.turbo_vae.encode(video, parallel=True)
        audio_latent = encode_audio(
            Audio(waveform=audio, sampling_rate=audio_rate), self.audio_encoder, self.audio_processor,
        )
        restored_video, restored_audio = denoise_one_step(
            self.transformer, video_latent, audio_latent, self.prompts,
            frame_rate, self.seed, self.condition_noise,
        )
        pixels = self.turbo_vae.decode(restored_video.to(torch.bfloat16), parallel=True)
        pixels = pixels[:, :, self.trim_frames:]
        if pixels.shape != video.shape:
            raise RuntimeError(f"Unexpected decoded video shape: {tuple(pixels.shape)}")
        waveform = decode_audio(restored_audio.to(torch.bfloat16), self.audio_decoder, self.vocoder)
        return Restoration(
            video=pixels, audio=waveform, video_latent_shape=tuple(video_latent.shape),
            audio_latent_shape=tuple(audio_latent.shape),
        )
