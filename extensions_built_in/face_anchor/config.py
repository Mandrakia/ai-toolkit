import os


class FaceAnchorConfig:
    """Parsed from the `face_anchor:` block of the training process config.

    Example YAML (inside config.process[i]):
        face_anchor:
          enabled: true
          identity_loss_weight: 0.1     # 0 disables; start 0.01-0.1
          tile_frac: 0.6                # face window as fraction of frame (data-derived: covers 99.3% of faces)
          min_cos: 0.2                  # skip gradient push on samples below this cos (hallucinated faces)
          yaw_gate: 50.0                # skip |yaw| > this (deg): strong profiles, alignment unreliable
          min_t: 0.0                    # timestep-ratio window the anchor fires in
          max_t: 1.0
          clean_cos_target: true        # target each image's own clean cos, not 1.0 (pose-fair)
          cache_path: ".face_anchor_cache.pt"   # built by caching.build_cache (preflight, needs insightface)
          arcface_onnx: "~/.insightface/models/buffalo_l/w600k_r50.onnx"  # placed by insightface during preflight
          taef2_path: null                        # optional; if unset, auto-downloads madebyollin/taef2 (cached)
    """

    def __init__(self, **kw):
        self.enabled: bool = kw.get("enabled", False)
        self.identity_loss_weight: float = float(kw.get("identity_loss_weight", 0.0))
        self.tile_frac: float = float(kw.get("tile_frac", 0.6))
        self.min_cos: float = float(kw.get("min_cos", 0.2))
        self.yaw_gate: float = float(kw.get("yaw_gate", 50.0))
        self.min_t: float = float(kw.get("min_t", 0.0))
        self.max_t: float = float(kw.get("max_t", 1.0))
        self.clean_cos_target: bool = kw.get("clean_cos_target", True)
        # Target the generated face is pushed toward:
        #   per_image = THIS image's own GT embedding (real, angle-matched — recommended for training,
        #               avoids pushing toward a fictional centroid average and impossible profile targets)
        #   centroid  = dataset/group mean embedding (GLT-style; pose-fair via clean_cos)
        #   blend     = normalize(target_blend*per_image + (1-target_blend)*centroid)
        self.identity_target: str = kw.get("identity_target", "per_image")
        self.target_blend: float = float(kw.get("target_blend", 0.8))
        # ArcFace bias correction: subtract the mean embedding of random-noise crops from gen & ref
        # before cosine, so non-faces score ~0 instead of ArcFace's ~0.5 floor (cleaner gradient +
        # meaningful min_cos). Needs GT embeddings in the cache (rebuild it) — auto-disables if absent.
        self.bias_correction: bool = kw.get("bias_correction", True)
        self.bias_n: int = int(kw.get("bias_n", 200))
        self.cache_path: str = os.path.expanduser(kw.get("cache_path", ".face_anchor_cache.pt"))
        self.arcface_onnx: str = os.path.expanduser(
            kw.get("arcface_onnx", "~/.insightface/models/buffalo_l/w600k_r50.onnx"))
        # optional explicit path; if None, load_taef2 auto-downloads madebyollin/taef2 (HF cache, idempotent)
        _taef2 = kw.get("taef2_path", None)
        self.taef2_path = os.path.expanduser(_taef2) if _taef2 else None
        # the anchor is active only if explicitly enabled AND given a weight
        self.active: bool = self.enabled and self.identity_loss_weight > 0.0
