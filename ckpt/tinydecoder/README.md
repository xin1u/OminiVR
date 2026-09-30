# Optional legacy TinyDecoder

`taeltx2_3_wide.pth` is the decoder-only TAEHV checkpoint from the earlier
OmniVR release. Its implementation is bundled in
[`inference_tiny_decoder.py`](../../packages/omini-restore-trainer/scripts/inference_tiny_decoder.py).
Attribution and its MIT license are retained in [NOTICE](../../NOTICE).

The multistep entry point uses the full LTX video VAE by default. To select
the legacy decoder explicitly:

```bash
hf download xin1u/OmniVR tinydecoder/taeltx2_3_wide.pth --local-dir ckpt
# Add this option to the multistep command:
# --tiny_decoder ckpt/tinydecoder/taeltx2_3_wide.pth
```

This checkpoint cannot replace the Flash Turbo VAE: Flash requires both an
UltraTinyVAE encoder and decoder matching the bundled `configs/turbo_vae.json`.
