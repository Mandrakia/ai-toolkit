"""The identity anchor loss: x0_pred -> fixed face tile decode -> aligned warp -> ArcFace -> 1-cos."""
import numpy as np
import torch
import torch.nn.functional as F

from .config import FaceAnchorConfig
from .caching import FaceAnchorCache
from .geometry import estimate_norm, fixed_window_latent, warp_batch
from .models import load_arcface, arcface_embed, load_taef2, taef2_decode


class FaceAnchor:
    def __init__(self, cfg: FaceAnchorConfig, device, dtype):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.arcface = load_arcface(cfg.arcface_onnx, device)   # fp32, frozen (loaded once)
        self.taef2 = load_taef2(cfg.taef2_path, device, dtype)  # tiny VAE decoder, frozen (loaded once)
        self.cache = FaceAnchorCache.load(cfg.cache_path)
        # --- precompute everything that never changes across steps ---
        # centroids resident on GPU (else a CPU->GPU copy per sample per step)
        self.centroids = {k: t.to(device).float() for k, t in self.cache._centroids.items()}
        # fixed 112x112 warp destination grid, built once and reused every step
        out = 112
        ys, xs = torch.meshgrid(torch.arange(out, device=device, dtype=torch.float32),
                                torch.arange(out, device=device, dtype=torch.float32), indexing="ij")
        self._dst = torch.stack([xs.flatten(), ys.flatten(), torch.ones(out * out, device=device)], 0)
        # per-(image, latent-size) geometry memo: skimage estimate_norm + window run once per key
        self._geom_cache = {}

        # --- ArcFace bias correction (subtract mean-noise embedding so non-faces score ~0) ---
        self.bias = None
        have_emb = any("embedding" in m for m in self.cache.images.values())
        if cfg.bias_correction:
            if not have_emb:
                print("[face_anchor] bias_correction on but cache has no GT embeddings — disabled. "
                      "Rebuild the cache (caching.build_cache stores 'embedding').")
            else:
                with torch.no_grad():
                    noise = torch.rand(cfg.bias_n, 3, 112, 112, device=device) * 255.0
                    self.bias = arcface_embed(self.arcface, noise).mean(0)   # (512,) mean of unit embeds

        # --- per-path effective target embedding + clean ceiling (mode-aware) ---
        # per_image: push toward THIS image's real embedding (angle-matched, no fictional average)
        # centroid : push toward the dataset/group mean (pose-fair via clean_cos)
        # blend    : normalize(b*per_image + (1-b)*centroid)
        self._target = {}   # path -> target embedding (GPU, unit-norm)
        self._clean = {}    # path -> loss ceiling (1.0 for per_image/blend; clean_cos for centroid)
        mode = cfg.identity_target
        for p, m in self.cache.images.items():
            c = self.centroids.get(m.get("dataset"))
            if c is None:
                continue
            e = m.get("embedding")
            e_t = torch.tensor(np.asarray(e, np.float32), device=device) if e is not None else None
            if mode == "per_image" and e_t is not None:
                tgt, clean = e_t, 1.0
            elif mode == "blend" and e_t is not None:
                b = cfg.target_blend
                tgt = F.normalize((b * e_t + (1.0 - b) * c).reshape(1, -1)).reshape(-1)
                clean = 1.0
            else:  # centroid (or per_image/blend with no GT embedding available)
                tgt = c
                clean = self._cos(e_t, c) if (cfg.clean_cos_target and e_t is not None) else 1.0
            self._target[p] = tgt
            self._clean[p] = max(0.05, clean)
        print(f"[face_anchor] target={mode} (bias_correction={'on' if self.bias is not None else 'off'}); "
              f"{len(self._target)} images")

    def _cos(self, a, b):
        """Cosine in the (optionally bias-corrected) space, on 1D embeddings."""
        if self.bias is not None:
            a, b = a - self.bias, b - self.bias
        return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())

    def _geom(self, path, meta, B, lh, lw):
        """Window offset + warp matrix for (image, latent-size). Deterministic -> memoized."""
        key = (path, lh, lw)
        g = self._geom_cache.get(key)
        if g is None:
            kps_norm = np.asarray(meta["kps_norm"], np.float32)
            kps_px = kps_norm * np.array([lw * 8, lh * 8], np.float32)
            cx, cy = kps_norm.mean(0) * np.array([lw, lh], np.float32)
            lx0, ly0 = fixed_window_latent((cx, cy), B, lh, lw)
            M = estimate_norm(kps_px - np.array([lx0 * 8, ly0 * 8], np.float32))
            g = (lx0, ly0, torch.from_numpy(M).to(self.device))
            self._geom_cache[key] = g
        return g

    def compute_loss(self, x0_pred: torch.Tensor, file_paths, t_ratio: torch.Tensor):
        """x0_pred: (n,32,lh,lw) recovered clean latent (grad on). t_ratio: (n,) in [0,1].
        Returns (scalar loss, metrics dict). Assumes one bucket size per batch.
        """
        cfg = self.cfg
        n, _, lh, lw = x0_pred.shape
        B = int(np.clip(round(cfg.tile_frac * min(lh, lw)), 8, min(lh, lw)))
        tr = t_ratio.detach().float().cpu().tolist()   # one GPU->CPU sync, not one per sample

        # --- gather per-sample fixed-size latent tiles + (memoized) warp matrices ---
        tiles, Ms, refs, cleans, tw = [], [], [], [], []
        for i, path in enumerate(file_paths):
            meta = self.cache.get(path)
            if meta is None:
                continue
            if abs(meta["yaw"]) > cfg.yaw_gate:                 # strong profile: alignment unreliable
                continue
            if tr[i] < cfg.min_t or tr[i] > cfg.max_t:
                continue
            ref = self._target.get(path)                        # mode-aware target (per_image/centroid/blend)
            if ref is None:
                continue
            lx0, ly0, M = self._geom(path, meta, B, lh, lw)     # skimage runs once per (image,res)
            tiles.append(x0_pred[i:i + 1, :, ly0:ly0 + B, lx0:lx0 + B])
            Ms.append(M)
            refs.append(ref)
            cleans.append(self._clean.get(path, 1.0))           # 1.0 for per_image/blend; clean_cos for centroid
            tw.append(tr[i])

        if not tiles:
            return x0_pred.new_zeros(()), {"id_n": 0}

        # --- batched decode -> warp -> ArcFace (grad flows back to x0_pred -> noise_pred -> LoRA) ---
        tile_img = taef2_decode(self.taef2, torch.cat(tiles, 0))         # (k,3,B*8,B*8) [0,255]
        aligned = warp_batch(tile_img, torch.stack(Ms), out=112, dst=self._dst)  # (k,3,112,112)
        emb = arcface_embed(self.arcface, aligned)                       # (k,512) normalized
        ref = torch.stack(refs)                                          # (k,512)
        if self.bias is not None:
            cos = F.cosine_similarity(emb - self.bias, ref - self.bias, dim=-1)
        else:
            cos = (emb * ref).sum(-1)

        clean = torch.tensor(cleans, device=self.device)
        weight = torch.tensor(tw, device=self.device)                    # timestep weighting
        # pose-fair shortfall vs each image's own clean cos (perceptual-repo style)
        per = torch.clamp(1.0 - cos / clean, min=0.0)
        # don't push gradient on hallucinated faces (cos below threshold) — gate on detached value
        push = (cos.detach() > cfg.min_cos).float()
        loss = (per * weight * push).sum() / push.sum().clamp_min(1.0)
        return loss, {"id_n": len(tiles), "id_push": int(push.sum().item()),
                      "id_cos": float(cos.detach().mean()), "id_B": B}
