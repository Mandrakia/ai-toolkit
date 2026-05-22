# Face Identity Anchor

A differentiable **ArcFace identity loss** for **Flux-2 Klein** character LoRAs, added alongside the
diffusion loss so the character converges to a recognizable identity faster. Inspired by the perceptual
identity loss in `ai-toolkit-perceptual` (BuffaloBuffalo). The identity cache is built **automatically
per run** (see below); by default each generated face is pushed toward **its own** real embedding
(`identity_target: per_image`), not a shared centroid.

Per step, for each kept face-bearing sample (kept = `sigma ∈ [min_t, max_t]` and `|yaw| < yaw_gate`):

```
noise_pred ──► x0 = noisy_latents − sigma·noise_pred        (flow-matching clean latent, grad on)
            ──► crop the face's fixed-size latent tile (tile_frac of the frame)
            ──► taef2 decode (tiny VAE)  ──►  RGB tile  ──► landmark-aligned 112 warp
            ──► ArcFace (onnx2torch w600k_r50, frozen)  ──►  512-d embedding
            ──► cos(emb, target)        target = this image's own embedding (per_image, default)
            ──► loss = clamp(1 − cos/target_ceiling, 0) · sigma   (pushed only where cos > min_cos)
loss += identity_loss_weight · mean(loss over kept samples)
```

## Why these choices

- **Differentiable ArcFace via onnx2torch** of insightface `w600k_r50.onnx`: numerically identical to
  insightface (cos 1.0) and cheap (~18 ms / 0.6 GB for a bs8 fwd+bwd). Frozen.
- **Target = `per_image` (default)**: push each face toward THAT image's own real embedding, not a
  fictional dataset-average centroid (which is nobody, and an impossible target for off-frontal shots).
  `centroid` / `blend` modes exist for the GLT-style shared-centroid case.
- **`target_ceiling` (0.9)**: cos=1.0 is unreachable (taef2 + warp + bias-correction all lower it), so
  targeting it over-corrects toward a point the pipeline can't reach — cap the loss at the achievable
  plateau.
- **`min_t` / `max_t` window (default 0.5–0.75) — the lighting lever**: ArcFace is illumination-invariant,
  so when the anchor fires at low noise (where the model renders shading) it teaches the LoRA to make
  faces flat — no shadows, no play of light. `min_t` keeps it in the high-noise *geometry* regime; `max_t`
  drops the top noise band where `x0_pred` is unrecognizable mush (no usable identity). The recognizable/
  shaded zone is **framing-driven** (face area ÷ photo area, not pose), so the defaults are set to the
  tightest-framed worst case — run `probe_min_t.py` to re-measure for a dataset. NB: these are `sigma`
  cuts and assume **no resolution-dependent timestep shift** — exact for `timestep_type`
  sigmoid/weighted/linear, but NOT for shift/flux_shift/lumina2_shift.
- **`bias_correction` (default on)**: subtract the mean ArcFace embedding of random-noise crops from both
  sides before the cosine, so non-faces score ~0 instead of ArcFace's ~0.5 floor (cleaner gradient +
  meaningful `min_cos`). Uses the GT embeddings the cache stores.
- **Landmark alignment (not bbox crop)**: a `grid_sample` warp from insightface's `estimate_norm`
  reproduces `norm_crop` to cos 0.9998+; roll is irrelevant once aligned, only out-of-plane yaw/pitch
  remain. Plus a hard `yaw_gate` (~50°) skip for strong profiles. Avoids double-penalizing off-frontal faces.
- **taef2 tiny VAE for the per-step decode**: the full Flux-2 VAE OOMs full-frame at bs>1, so the loss
  decodes a fixed face tile with `madebyollin/taef2` (`AutoencoderTiny(latent_channels=32)`) — cheap and
  identity-faithful (ArcFace is color-robust). Its **colorimetry is only approximate**, which is why the
  `probe_min_t` diagnostic decodes with the real VAE for faithful color.
- **Fixed-size face tile (`tile_frac=0.6`)**: latent *resize* destroys identity, but a fixed *window*
  (constant fraction of the frame, no resize) preserves it. 0.6 covers ~99% of the real face-size
  distribution. Fixed shape ⇒ predictable VRAM, no step-30 OOM, compile-friendly.

## Enable it

Set the process `type: face_anchor_trainer`. The defaults are tuned so an **empty (or absent)
`face_anchor:` block already runs it well** — the cache builds automatically at the start of the run
(no offline preflight). Override only what you need:

```yaml
face_anchor:
  enabled: true                # default true once you select face_anchor_trainer (0-weight or false disables)
  identity_loss_weight: 0.1    # default 0.1
  # defaults, shown for reference — usually leave them:
  #   identity_target: per_image | target_ceiling: 0.9 | bias_correction: true
  #   min_t: 0.5 | max_t: 0.75  (the lighting window; framing-specific — see probe_min_t.py)
  #   tile_frac: 0.6 | min_cos: 0.2 | yaw_gate: 50
  #   auto_cache: true  (builds the cache; no cache_path needed)
  #   taef2_path: null  (auto-downloads madebyollin/taef2 to the HF cache)
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

## TODO / next

- **Adaptive per-image `min_t`/`max_t`**: the window is framing-driven, so a fixed cut (set to the
  tightest-framed worst case) under-uses the anchor on wide/medium-framed shots — they fire mostly in
  their own mush band. Make the threshold `f(face_frac)` (face area ÷ photo area, already in the cache
  via `bbox_norm`), derivable from the flow-matching shift formula (face token count). Keeps the anchor
  uniform across a deliberately-diverse dataset without curating it.
- **Validate on a real run**: confirm the anchor improves held-out identity without flattening lighting
  (now that `min_t`/`max_t` are tuned); tune `identity_loss_weight`.
- **Cache crop offset**: train-side kps mapping assumes aspect-preserving resize only. If the data
  pipeline center-crops, store and apply the crop transform (FileItemDTO).
- **Logging**: surface `id_n / id_cos / id_B` into the training log + progress bar (currently console only).
- Optional levers: skip decode for reg/no-face samples; grad-checkpoint the tiny decode; sub-batch the anchor.
