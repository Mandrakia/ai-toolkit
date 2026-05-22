# Face Identity Anchor

A differentiable **ArcFace identity loss** added alongside the diffusion loss, to keep a
character LoRA recognizable. Inspired by `ai-toolkit-perceptual` (BuffaloBuffalo), adapted
for **Flux-2 Klein** and for reusing a precomputed identity **centroid** (e.g. from GridLoraTester).

Per step, for each face-bearing sample:

```
noise_pred ──► x0 = noisy_latents − sigma·noise_pred      (flow-matching clean latent, grad on)
            ──► crop the face's fixed-size latent tile (tile_frac of the frame)
            ──► taef2 decode (tiny VAE)  ── grad ──►  RGB tile
            ──► landmark-aligned warp to 112×112  (estimate_norm + grid_sample)
            ──► ArcFace (onnx2torch w600k_r50, frozen)  ──►  512-d embedding
            ──► loss = clamp(1 − cos(emb, centroid)/clean_cos, 0) · t_ratio
loss += identity_loss_weight · mean(loss over kept samples)
```

## Why these choices (all smoke-tested 2026-05-21)

- **Differentiable ArcFace via onnx2torch** of insightface `w600k_r50.onnx`: numerically identical
  to insightface (cos 1.0), ~18 ms / 0.6 GB for bs8 fwd+bwd. Same weights as a GLT centroid → the
  centroid is reusable verbatim.
- **Landmark alignment (not bbox crop)**: a `grid_sample` warp from insightface's `estimate_norm`
  reproduces `norm_crop` to cos 0.9998+, profile included. Roll is confirmed irrelevant once aligned
  (partial corr +0.08); only out-of-plane yaw/pitch remain. Alignment avoids double-penalizing
  off-frontal faces.
- **taef2 tiny VAE** (`madebyollin/taef2`, 32-ch): loads into `diffusers.AutoencoderTiny(latent_channels=32)`
  (decoder exact, missing=0). Preserves identity (roundtrip cos ≈ full VAE within ~0.02-0.04) at
  ~13× lower cost than the full Flux-2 VAE (which OOMs full-frame at bs>1).
- **Fixed-size face tile (`tile_frac`)**: latent **resize** destroys identity (don't), but a fixed
  *window* (constant fraction of the frame, no resize) preserves it. `tile_frac=0.6` covers **99.3%**
  of the real face-size distribution (GLT: median face = 28% of frame, p95 = 51%). Fixed shape ⇒
  predictable VRAM, no step-30 OOM, compile-friendly. ~0.65 GB/sample at 1024 (bs8 ≈ 5 GB).
- **No formula for pose**: per-image `clean_cos` is the exact pose-fair target (a fitted
  deltaSim=f(yaw,pitch) is not transferable across characters — R² 0.07→0.81). Plus a hard
  `yaw_gate` (~50°) skip for strong profiles, where alignment is unreliable and data is sparse.

## Enable it

1. Build the cache once (offline, needs insightface — run in the GridLoraTester venv).
   GLT prep: datasets curated; for a multi-folder character make a GLT **group** (its group
   centroid lands in glt.db). Find a group id: `sqlite3 glt.db "SELECT id,name,paths_json FROM dataset_groups"`.

   ```bash
   # group (N folders = one identity, uses the GLT group centroid):
   PYTHONPATH=/path/to/ai-toolkit /path/to/GridLoraTester/.venv/bin/python \
     -m extensions_built_in.face_anchor.caching \
     --glt-db /path/to/glt.db --group <id> --key <name> --out /abs/cache.pt \
     --dirs /datasets/folderA /datasets/folderB ...
   # folders (each folder = its own identity):
   ... -m extensions_built_in.face_anchor.caching --glt-db /path/glt.db --out /abs/cache.pt --dirs /datasets/michel
   ```
   It detects faces, picks the target face (max cos to centroid), and stores per image
   kps/bbox/yaw/clean_cos + the target **embedding** (needed for bias-correction) + the centroid.
   Re-run when the datasets or centroid change. The training venv needs no insightface — only the .pt.
2. In your training config, set the process `type: face_anchor_trainer` and add:

```yaml
face_anchor:
  enabled: true
  identity_loss_weight: 0.1
  tile_frac: 0.6
  min_cos: 0.2
  yaw_gate: 50.0
  cache_path: "/abs/path/.face_anchor_cache.pt"
  # taef2_path: optional. If unset, madebyollin/taef2 auto-downloads to the HF cache (idempotent).
```

Models needed at train time:
- **taef2** — auto-downloads on first run (HF cache). Override with `taef2_path` only if you want a local copy.
- **ArcFace `w600k_r50.onnx`** — placed under `~/.insightface/models/buffalo_l/` by insightface during
  the cache preflight, so it's already there when you train. (No insightface needed in the training venv.)

Keep the anchor **eager / outside any `torch.compile` region** (data-dependent gating + skimage).

## TODO (skeleton → production)

- **Cache crop offset**: train-side kps mapping assumes aspect-preserving resize only. If the data
  pipeline center-crops, store and apply the crop transform (FileItemDTO) — otherwise the window is
  slightly off on cropped images.
- **Logging**: surface `id_n / id_cos / id_B` into the training log + progress bar.
- **ArcFace bias correction** (subtract mean noise-crop embedding) for cleaner non-face scores.
- **Validate on a real run**: confirm the anchor improves held-out identity without burning in / hurting
  pose generalization; tune `identity_loss_weight` (start 0.01–0.1).
- Optional levers: skip decode for reg/no-face samples; grad-checkpoint the tiny decode; sub-batch the anchor.
