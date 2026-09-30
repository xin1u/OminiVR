# LTX-Core for Omnivr-Flash

This directory contains the shared LTX model definitions and runtime used by
[Omnivr-Flash and the original multi-step OmniVR path](../../README.md).
The inference launchers add `src/` to the
Python search path automatically.

The Flash path uses:

- The joint audio/video transformer and its additive conditioning projections.
- Audio VAE encoding and decoding, audio preprocessing, and the vocoder.
- Safetensors checkpoint loading and configuration mapping.
- Latent shapes, patchifiers, noise generation, and the Euler diffusion step.

Video encoding and decoding use the bundled UltraTinyVAE implementation in
`packages/Omnivr-Flash/src/omnivr_flash/turbo_vae.py`. Prompt embeddings are
loaded from a cache by Omnivr-Flash.

The shared model library also retains the upstream video VAE, text encoders,
conditioning utilities, and optional inference backends to preserve its
internal module interfaces. Omnivr-Flash uses the settings described in the
[root README](../../README.md).

Install dependencies from the root `requirements.txt` and run the root
`infer.py`. No training package or editable installation is required. The
inference-specific component loaders live in
`packages/Omnivr-Flash/src/omnivr_flash/model_loader.py`. The original multi-step
path uses `packages/omini-restore-trainer/scripts/inference_model_loader.py`.

LTX components are covered by the [LTX-2 Community License](../../LICENSE).
