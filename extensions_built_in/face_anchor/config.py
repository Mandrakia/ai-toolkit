import os


class FaceAnchorConfig:
    """Parsed from the `face_anchor:` block of the training process config.

    Defaults are tuned so that selecting `type: face_anchor_trainer` with NO face_anchor block already
    runs the anchor at the recommended settings. Every key below shows its default; override only what
    you need. Disable with `enabled: false` or `identity_loss_weight: 0`.

    Example YAML (inside config.process[i]):
        face_anchor:
          enabled: true                 # default true (the trainer's whole point)
          identity_loss_weight: 0.1     # default 0.1; tune; 0 disables
          identity_target: per_image    # per_image (default) | centroid | blend
          target_ceiling: 0.9           # cos plateau cap (per_image/blend); avoids over-correcting to an unreachable target
          tile_frac: 0.6                # face window as fraction of frame (covers 99.3% of faces)
          min_cos: 0.2                  # skip gradient push on samples below this cos (hallucinated faces)
          yaw_gate: 50.0                # skip |yaw| > this (deg): strong profiles, alignment unreliable
          min_t: 0.5                    # only fire in high-noise (geometry) regime — else faces come out flat (no shadows/light play)
          max_t: 0.75                   # above this x0_pred is unrecognizable mush (no usable identity)
          bias_correction: true         # subtract mean-noise embedding so non-faces score ~0
          # Cache is automatic (auto_cache: true): a per-dataset sidecar (<dataset>/face_anchor.pt) is
          # built+aggregated into <output>/<run_name>/anchor_cache.pt, rebuilt only when the data changes.
          # Set auto_cache: false + cache_path to use a hand-built (e.g. GLT) cache verbatim.
          # buffalo_l ONNX (det_10g + w600k_r50) and taef2 auto-download from HF if not already on disk
          # (no insightface / onnxruntime needed); override det_onnx / arcface_onnx / taef2_path to pin them.
    """

    def __init__(self, **kw):
        # Default ON: picking `type: face_anchor_trainer` already signals intent to use the anchor —
        # so an empty (or absent) face_anchor block runs it at the recommended weight. Disable with
        # `enabled: false` or `identity_loss_weight: 0`.
        self.enabled: bool = kw.get("enabled", True)
        self.identity_loss_weight: float = float(kw.get("identity_loss_weight", 0.1))
        self.tile_frac: float = float(kw.get("tile_frac", 0.6))
        self.min_cos: float = float(kw.get("min_cos", 0.2))
        self.yaw_gate: float = float(kw.get("yaw_gate", 50.0))
        # min_t 0.5: only fire in the high-noise (geometry) regime. ArcFace is illumination-invariant,
        # so letting the anchor act at low noise — where shading is rendered — made the generated faces
        # come out flat: no shadows, no play of light at all. Keeping it high-noise preserves the
        # rendered shading. This is THE flat-lighting lever.
        # CAVEAT — does NOT handle the resolution-dependent timestep SHIFT. min_t/max_t gate on
        # t01 = timesteps/1000 taken as the noise fraction (sigma). That's exact for timestep_type
        # sigmoid/weighted/linear (no shift) → min_t is the same sigma on every resolution bucket. With
        # timestep_type shift/flux_shift/lumina2_shift, t01 is the *resolution-shifted* sigma, so a fixed
        # min_t cuts at a different schedule position per bucket (e.g. base-0.5 lands at sigma ~0.65 @512
        # vs ~0.76 @1024). Not corrected — only use shift modes with this in mind.
        self.min_t: float = float(kw.get("min_t", 0.5))
        # max_t 0.75: above this, x0_pred is unrecognizable mush (no usable identity) so the anchor's
        # gradient there is wasted/noisy. Set from the probe_min_t diagnostic (drop the top ~2 of 8 sigma
        # columns). 0.75 is the SAFE choice across framings — tight close-ups stay recognizable a bit
        # higher (could allow ~0.875), wide shots degrade earlier; 0.75 covers the common medium range.
        self.max_t: float = float(kw.get("max_t", 0.75))
        self.clean_cos_target: bool = kw.get("clean_cos_target", True)
        # Target the generated face is pushed toward:
        #   per_image = THIS image's own GT embedding (real, angle-matched — recommended for training,
        #               avoids pushing toward a fictional centroid average and impossible profile targets)
        #   centroid  = dataset/group mean embedding (GLT-style; pose-fair via clean_cos)
        #   blend     = normalize(target_blend*per_image + (1-target_blend)*centroid)
        self.identity_target: str = kw.get("identity_target", "per_image")
        self.target_blend: float = float(kw.get("target_blend", 0.8))
        # per_image/blend loss ceiling. cos=1.0 is UNREACHABLE (taef2+warp+arcface degrade it, and
        # bias-correction lowers same-pair cos), so targeting it over-corrects — the loss keeps pushing
        # toward a target the pipeline can't hit. Cap at the achievable plateau (tune to the id_cos you
        # see in the logs). (Flat-lighting is min_t's job, not this.)
        self.target_ceiling: float = float(kw.get("target_ceiling", 0.9))
        # ArcFace bias correction: subtract the mean embedding of random-noise crops from gen & ref
        # before cosine, so non-faces score ~0 instead of ArcFace's ~0.5 floor (cleaner gradient +
        # meaningful min_cos). Needs GT embeddings in the cache (rebuild it) — auto-disables if absent.
        self.bias_correction: bool = kw.get("bias_correction", True)
        self.bias_n: int = int(kw.get("bias_n", 200))
        self.cache_path: str = os.path.expanduser(kw.get("cache_path", ".face_anchor_cache.pt"))
        # Auto-cache (default): build a per-dataset sidecar (<dataset>/<cache_filename>) keyed to the
        # image set, aggregate them into <output>/run_name/anchor_cache.pt (one global centroid =
        # one identity), and point the anchor at that aggregate. Stale caches rebuild on data change.
        # Set false to use a hand-built cache_path verbatim (legacy / GLT-built).
        self.auto_cache: bool = kw.get("auto_cache", True)
        self.cache_filename: str = kw.get("cache_filename", "face_anchor.pt")  # per-dataset sidecar name
        self.identity_key: str = kw.get("identity_key", "global")              # run-level centroid key
        self.arcface_onnx: str = os.path.expanduser(
            kw.get("arcface_onnx", "~/.insightface/models/buffalo_l/w600k_r50.onnx"))
        self.det_onnx: str = os.path.expanduser(
            kw.get("det_onnx", "~/.insightface/models/buffalo_l/det_10g.onnx"))  # SCRFD for auto_cache
        # optional explicit path; if None, load_taef2 auto-downloads madebyollin/taef2 (HF cache, idempotent)
        _taef2 = kw.get("taef2_path", None)
        self.taef2_path = os.path.expanduser(_taef2) if _taef2 else None
        # active unless explicitly turned off (enabled: false) or zero-weighted (identity_loss_weight: 0)
        self.active: bool = self.enabled and self.identity_loss_weight > 0.0
