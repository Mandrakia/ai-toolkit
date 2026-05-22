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

Set the process `type: face_anchor_trainer` and add a `face_anchor:` block. The cache is built
**automatically at the start of the run** — no offline preflight needed:

```yaml
face_anchor:
  enabled: true
  identity_loss_weight: 0.1
  tile_frac: 0.6
  min_cos: 0.2
  yaw_gate: 50.0
  # auto_cache: true  (default) — see below; no cache_path needed
  # taef2_path: optional. If unset, madebyollin/taef2 auto-downloads to the HF cache (idempotent).
```

### Auto-cache (default) — built per-dataset + aggregated per-run

On `hook_before_train_loop` (before the first step), the trainer mirrors the latent-cache pattern:

1. **Per dataset** → writes a sidecar `<dataset_folder>/face_anchor.pt`, keyed by a **content
   signature** of the image set (sorted realpath + `size:mtime` + `CACHE_VERSION`). On every run it
   re-checks the signature: unchanged ⇒ reuse, changed (image added/removed/edited) ⇒ rebuild only
   that dataset.
2. **Per run** → aggregates the sidecars into `<output>/<run_name>/anchor_cache.pt`: unions all face
   entries and computes **one global centroid** (every dataset in a run is treated as one identity;
   reg datasets and video/audio datasets are skipped). Rebuilt only if any sidecar's signature, or
   the set of datasets, changed. The anchor then loads this aggregate.

Detection/embedding run **in-venv** via torch SCRFD + onnx2torch ArcFace — **no insightface, no
onnxruntime, no pre-populated `~/.insightface`**. All model files auto-resolve:
- **buffalo_l ONNX** (`det_10g.onnx` SCRFD + `w600k_r50.onnx` ArcFace): used from
  `~/.insightface/models/buffalo_l/` if already there (e.g. insightface installed), else
  auto-downloaded from HF `public-data/insightface` (byte-identical) into the HF cache. Override the
  on-disk lookup with `det_onnx:` / `arcface_onnx:`.
- The `onnx2torch.convert` (~3.2s) is cached to `~/.cache/face_anchor/onnx2torch/*.pt` (~36x faster
  reload, keyed by torch+onnx2torch version + onnx size:mtime), so stop/start and later trainings skip it.
- **taef2** auto-downloads on first run (HF cache); override with `taef2_path` for a local copy.

### Legacy / hand-built cache

Set `auto_cache: false` and point `cache_path:` at a `.pt` you built offline — e.g. to reuse a
**GridLoraTester** group centroid (run in the GLT venv, which has insightface):

```bash
# group (N folders = one identity, uses the GLT group centroid):
PYTHONPATH=/path/to/ai-toolkit /path/to/GridLoraTester/.venv/bin/python \
  -m extensions_built_in.face_anchor.caching \
  --glt-db /path/to/glt.db --group <id> --key <name> --out /abs/cache.pt \
  --dirs /datasets/folderA /datasets/folderB ...
# fully self-contained (no insightface), each dir its own identity unless --key groups them:
... -m extensions_built_in.face_anchor.caching --independent --out /abs/cache.pt --dirs /datasets/michel
```

Then `cache_path: "/abs/cache.pt"` + `auto_cache: false`.

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
