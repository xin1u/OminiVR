# TinyDecoder weights

`taeltx2_3_wide.pth` — decoder-only weights for the `TAEHV` module in
[`src/ominivr/tiny_decoder.py`](../../src/ominivr/tiny_decoder.py). Fast
latent→pixel decode path used in place of the full VAE decoder (~38MB vs.
loading the full video VAE decoder, and stays resident on GPU alongside the
transformer during inference).

Retrained on LTX-2.3's 128-channel video latent space; architecture and
training approach adapted from Ollin Boer Bohan's Seraena/TAESD (MIT
License) — see [`/NOTICE`](../../NOTICE) for full attribution.

## Download

```bash
huggingface-cli download xin1u/OminiVR-weights taeltx2_3_wide.pth --local-dir ckpt/tinydecoder
```

(Repo name is a placeholder until the weights upload is finalized — see the
main README for current status.)

If this file is absent, `scripts/infer.py` falls back to the full VAE decoder
automatically (slower, more memory).
