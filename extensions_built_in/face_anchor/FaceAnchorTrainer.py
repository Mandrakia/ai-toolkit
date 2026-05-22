"""DiffusionTrainer + differentiable ArcFace identity anchor.

Subclasses DiffusionTrainer (the UI/API trainer, itself an SDTrainer) so the full
diffusion_trainer behaviour — UI logger, sqlite, sampling, differential guidance — is kept;
we only add the identity loss by overriding calculate_loss.

Usage: set the process type to `face_anchor_trainer` and add a `face_anchor:` block (see config.py).
The anchor is computed eagerly and added to the diffusion loss before the single backward —
keep it OUTSIDE any torch.compile region (data-dependent gating + skimage).
"""
import os
import torch
from collections import OrderedDict
from typing import Union
from einops import rearrange

from toolkit.train_tools import get_torch_dtype
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer

from .config import FaceAnchorConfig


class FaceAnchorTrainer(DiffusionTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.face_anchor_config = FaceAnchorConfig(**self.get_conf("face_anchor", {}))
        self._face_anchor = None       # built lazily on first use (after devices/models are ready)
        self._fa_metrics = {}

    def _anchor(self):
        if self._face_anchor is None and self.face_anchor_config.active:
            from .anchor import FaceAnchor
            self._face_anchor = FaceAnchor(
                self.face_anchor_config,
                device=self.device_torch,
                dtype=get_torch_dtype(self.train_config.dtype),
            )
            print(f"\n[face_anchor] active: weight={self.face_anchor_config.identity_loss_weight} "
                  f"tile_frac={self.face_anchor_config.tile_frac} yaw_gate={self.face_anchor_config.yaw_gate}")
        return self._face_anchor

    def hook_before_train_loop(self):
        # datasets + dataloader exist by now (built in run() right before this hook); the model is
        # loaded but training hasn't started. Build/validate the anchor cache here, before _anchor()
        # loads it on the first calculate_loss.
        super().hook_before_train_loop()
        cfg = self.face_anchor_config
        if cfg.active and cfg.auto_cache:
            self._prepare_anchor_cache()

    def _prepare_anchor_cache(self):
        """Build per-dataset sidecars (<dataset>/<cache_filename>) + aggregate them into
        <save_root>/anchor_cache.pt, then point the anchor at the aggregate. Each level rebuilds only
        if its data changed (signature check). All datasets in a run = one identity (one centroid)."""
        from toolkit.data_loader import get_dataloader_datasets
        from . import caching
        cfg = self.face_anchor_config
        if self.data_loader is None:
            print("[face_anchor] no data_loader; auto_cache skipped, using cache_path as-is")
            return
        specs = []   # (key, sidecar_dir, image_paths)
        for ds in get_dataloader_datasets(self.data_loader):
            if getattr(ds, "is_video", False) or getattr(ds, "is_audio_model", False):
                continue  # the anchor is image-only
            paths = sorted({fi.path for fi in ds.file_list})  # dedupe num_repeats
            if not paths:
                continue
            base = ds.dataset_path
            sidecar_dir = base if os.path.isdir(base) else os.path.dirname(base)
            key = os.path.basename(os.path.normpath(sidecar_dir)) or "dataset"
            specs.append((key, sidecar_dir, paths))
        if not specs:
            print("[face_anchor] no image datasets found; auto_cache skipped")
            return

        agg_path = os.path.join(self.save_root, "anchor_cache.pt")
        # main process builds/writes; others wait then read the finished files from disk
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                caching.prepare_run_cache(
                    specs, agg_path,
                    device=self.device_torch,
                    arcface_onnx=cfg.arcface_onnx, det_onnx=cfg.det_onnx,
                    cache_filename=cfg.cache_filename, global_key=cfg.identity_key,
                )
        cfg.cache_path = agg_path
        print(f"[face_anchor] anchor cache: {agg_path}")

    def _to_vae_latent(self, x0):
        """The flux2-klein custom VAE packs+normalizes the latent to 128ch (ps=2x2 + BatchNorm),
        so the trainer works in that space. taef2 wants the raw 32ch VAE latent. Reproduce the
        model's own decode head (inv_normalize -> 2x2 unpatch) — cheap + differentiable.
        For VAEs without this packing (32ch already), pass through unchanged.
        """
        vae = self.sd.vae
        bn = getattr(vae, "bn", None)
        if bn is None or x0.shape[1] != bn.num_features:
            return x0   # not the packed flux2 VAE (or already a raw latent)
        eps = getattr(vae, "bn_eps", 1e-4)
        s = torch.sqrt(bn.running_var + eps).view(1, -1, 1, 1).to(x0.device, x0.dtype)
        m = bn.running_mean.view(1, -1, 1, 1).to(x0.device, x0.dtype)
        z = x0 * s + m                                      # inverse BatchNorm
        pi, pj = getattr(vae, "ps", [2, 2])
        return rearrange(z, "b (c pi pj) i j -> b c (i pi) (j pj)", pi=pi, pj=pj)  # unpatch -> 32ch

    def calculate_loss(self, noise_pred, noise, noisy_latents, timesteps, batch,
                       mask_multiplier=1.0, prior_pred=None, **kwargs):
        loss = super().calculate_loss(
            noise_pred, noise, noisy_latents, timesteps, batch,
            mask_multiplier=mask_multiplier, prior_pred=prior_pred, **kwargs)

        anchor = self._anchor()
        if anchor is None:
            return loss

        # Flow-matching clean-latent recovery. ai-toolkit's add_noise uses t01 = timesteps/1000:
        #   noisy = (1 - t01)*x0 + t01*noise ,  velocity target = noise - x0
        #   => x0 = noisy - t01 * noise_pred      (matches SDTrainer's denoised_latents recovery)
        # NB: batch.sigmas is only populated in the loss_target='source' path, so we derive t01 here.
        t01 = (timesteps.float() / 1000.0).clamp(0.0, 1.0)
        sig = t01.view(-1, *([1] * (noisy_latents.ndim - 1))).to(noisy_latents.dtype)
        x0_pred = noisy_latents - sig * noise_pred
        x0_pred = self._to_vae_latent(x0_pred)   # flux2-klein: packed 128ch -> raw 32ch VAE latent
        file_paths = [fi.path for fi in batch.file_items]

        id_loss, metrics = anchor.compute_loss(x0_pred, file_paths, t01)
        self._fa_metrics = metrics
        step = getattr(self, "step_num", 0)
        cos = metrics.get("id_cos")
        cos_s = f"{cos:.3f}" if cos is not None else "n/a"
        if not getattr(self, "_fa_logged", False):
            self._fa_logged = True
            in_cache = sum(1 for p in file_paths if anchor.cache.get(p) is not None)
            print(f"\n[face_anchor] first step: x0={tuple(x0_pred.shape)} id_n={metrics.get('id_n')} "
                  f"id_cos={cos_s} | batch paths in cache: {in_cache}/{len(file_paths)} "
                  f"(cache has {len(anchor.cache.images)}); e.g. {file_paths[0]}")
        # concise periodic monitor every 20 steps (once per optimizer step despite grad-accum)
        if step % 20 == 0 and getattr(self, "_fa_last_log", -1) != step:
            self._fa_last_log = step
            print(f"\n[face_anchor] step {step}: id_n={metrics.get('id_n')} push={metrics.get('id_push')} "
                  f"id_cos={cos_s} t01_mean={float(t01.mean()):.2f}")
        return loss + self.face_anchor_config.identity_loss_weight * id_loss
