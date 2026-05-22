"""Throwaway diagnostic: find a good face_anchor `min_t` WITHOUT paying for training runs.

Question it answers: at which noise level (sigma = t01 = timesteps/1000) does YOUR model start
rendering face shading/lighting? Below that, the illumination-invariant ArcFace anchor flattens it
(no shadows / no play of light); above it, the anchor only shapes geometry → safe. So set min_t just
above the shading-onset sigma.

How: it reuses the real trainer end-to-end (model load, latent encode, the exact forward, the anchor
cache) by monkeypatching `FaceAnchorTrainer.calculate_loss` AT RUNTIME — the extension files are NOT
modified. The patch recovers the genuine x0_pred (= noisy_latents - sigma*noise_pred, the packed 128ch
model latent) and decodes it with the REAL Flux-2 VAE (sd.decode_latents) for faithful color — taef2
(used by the live anchor) is great for ArcFace identity but only approximates colorimetry. The face is
landmark-aligned to 112 (same warp as the anchor) and we record, per sample:
    (sigma, cos-to-target, macro luminance-std of the aligned face)
`lr` is forced to 0 (weights never change) and saving is disabled. The trainer auto-resumes the latest
checkpoint of the config's `name`; to probe the BASE model (a flattened checkpoint can't show where the
model naturally renders shading), run against a config with a FRESH job name so it starts from step 0.
`timestep_type` is forced to `linear` for uniform sigma. One model load + a few hundred forwards, no training.

Outputs (in <save_root>/probe_min_t/):
  - contact_sheet.png : decoded face per (image, sigma bin) — eyeball where shadows appear (the truth)
  - curve.csv         : sigma_bin, mean_cos, mean_lum_std, n
  - curve.png         : the two curves (if matplotlib is present)

Usage:  python probe_min_t.py config/cha-global-p_miraclein_faceanchor.yaml [--calls 80] [--images 4] [--bins 10]
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image


class _StopProbe(Exception):
    """Raised once we've collected enough, to stop before the trainer trains/saves anything."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to the training YAML (the one you'd train with)")
    ap.add_argument("--calls", type=int, default=250, help="max forward micro-steps (= samples at bs=1)")
    ap.add_argument("--images", type=int, default=8, help="image ROWS in the sheet (use a SMALL dataset)")
    ap.add_argument("--bins", type=int, default=10, help="sigma buckets (columns) in [0,1]")
    ap.add_argument("--cell", type=int, default=110, help="contact-sheet cell size (px)")
    args = ap.parse_args()

    from toolkit.config import get_config
    from toolkit.job import get_job

    # --- load config and force probe-safe overrides (freeze + uniform sigma + no sampling) ---
    # NOTE: the trainer auto-resumes the latest checkpoint of the config's `name`. To probe the BASE
    # model (a checkpoint that already flattened lighting can't show where shading naturally renders),
    # run this against a config with a FRESH job name so it starts clean from step 0.
    cfg = get_config(args.config)
    procs = cfg["config"]["process"]
    for p in procs:
        tr = p.setdefault("train", {})
        tr["lr"] = 0.0                      # freeze: optimizer steps are no-ops -> weights never change
        tr["steps"] = 10_000_000            # never end naturally; we stop via the sentinel
        tr["disable_sampling"] = True
        tr["skip_first_sample"] = True
        tr["timestep_type"] = "linear"      # uniform sigma coverage (sigma meaning is mode-independent)
        if "ema_config" in tr:
            tr["ema_config"]["use_ema"] = False
        # a probe should never write checkpoints — disable saving (the sentinel also aborts before any save)
        sv = p.setdefault("save", {})
        sv["save_every"] = 10 ** 9
        sv["save_every_sec"] = 10 ** 12
        fa = p.setdefault("face_anchor", {})
        fa["enabled"] = True                # ensure the anchor (hence compute_loss) is active
        fa.setdefault("identity_loss_weight", 0.1)

    save_root = os.path.join(cfg["config"]["process"][0].get("training_folder", "output"),
                             cfg["config"].get("name", "probe"))
    out_dir = os.path.join(save_root, "probe_min_t")
    os.makedirs(out_dir, exist_ok=True)

    # --- collector + runtime monkeypatch on FaceAnchorTrainer.calculate_loss ---
    # We patch at the TRAINER level (not FaceAnchor.compute_loss) so we have the trainer's `self.sd`
    # and the packed 128ch latent: x0_pred = noisy_latents - sigma*noise_pred (same recovery the anchor
    # uses), decoded with the REAL Flux-2 VAE (sd.decode_latents) for faithful color — taef2 is great
    # for ArcFace identity but its colorimetry is only approximate. Face is then landmark-aligned to 112
    # (same warp as the anchor) for the cos/shading metric and the contact sheet.
    import torch.nn.functional as F
    from extensions_built_in.face_anchor.geometry import estimate_norm, warp_batch
    from extensions_built_in.face_anchor.models import arcface_embed
    from extensions_built_in.face_anchor.FaceAnchorTrainer import FaceAnchorTrainer

    rows = []            # list of (sigma, cos, lum_std, path) for the curves
    sheet = {}           # (path, sigma_bin) -> aligned 112 face : SAME image across a row
    seen = []            # ordered distinct paths that become rows (capped at --images)
    state = {"calls": 0}

    def _grid_full():
        return len(seen) >= args.images and all((p, b) in sheet for p in seen for b in range(args.bins))

    _orig_calc = FaceAnchorTrainer.calculate_loss

    def _probe_calc(self, noise_pred, noise, noisy_latents, timesteps, batch, **kw):
        try:
            with torch.no_grad():
                anchor = self._anchor()
                t01 = (timesteps.float() / 1000.0).clamp(0, 1)
                sig = t01.view(-1, *([1] * (noisy_latents.ndim - 1))).to(noisy_latents.dtype)
                imgs = self.sd.decode_latents(noisy_latents - sig * noise_pred)   # packed -> (n,3,H,W)
                imgs = (imgs.float().clamp(-1, 1) + 1.0) * (255.0 / 2.0)          # [-1,1] -> [0,255] RGB
                _, _, H, W = imgs.shape
                paths = [fi.path for fi in batch.file_items]
                tr = t01.detach().float().cpu().tolist()
                for i, path in enumerate(paths):
                    meta = anchor.cache.get(path)
                    ref = anchor._target.get(path)
                    if meta is None or ref is None:
                        continue
                    kps = np.asarray(meta["kps_norm"], np.float32) * np.array([W, H], np.float32)
                    M = torch.from_numpy(estimate_norm(kps)).unsqueeze(0).to(imgs.device)
                    aligned = warp_batch(imgs[i:i + 1], M, out=112)              # (1,3,112,112) [0,255]
                    emb = arcface_embed(anchor.arcface, aligned)[0]
                    if anchor.bias is not None:
                        cos = float(torch.cosine_similarity(
                            (emb - anchor.bias)[None], (ref - anchor.bias)[None]).item())
                    else:
                        cos = float((emb * ref).sum().item())
                    lum = 0.299 * aligned[0, 0] + 0.587 * aligned[0, 1] + 0.114 * aligned[0, 2]
                    lum_std = float(F.avg_pool2d(lum[None, None], 14, 14).std().item())  # macro 8x8 shading
                    sigma = tr[i]
                    rows.append((sigma, cos, lum_std, path))
                    # contact sheet: row = THIS image (pose-normalized 112), column = sigma bin
                    if path not in seen and len(seen) < args.images:
                        seen.append(path)
                    b = min(args.bins - 1, int(sigma * args.bins))
                    if path in seen and (path, b) not in sheet:
                        sheet[(path, b)] = aligned[0].clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        except Exception as e:
            print(f"[probe] harvest failed (continuing): {e}")

        state["calls"] += 1
        if state["calls"] % 10 == 0:
            print(f"[probe] {state['calls']}/{args.calls} micro-steps, {len(rows)} samples, "
                  f"sheet {len(sheet)}/{len(seen) * args.bins} cells ({len(seen)} image rows)")
        if state["calls"] >= args.calls or _grid_full():
            raise _StopProbe()
        # let the real loss/backward run (lr=0 -> no update, no save) to keep the loop alive
        return _orig_calc(self, noise_pred, noise, noisy_latents, timesteps, batch, **kw)

    FaceAnchorTrainer.calculate_loss = _probe_calc

    # --- run the real trainer until the sentinel fires ---
    print(f"[probe] loading model + building anchor cache via the trainer "
          f"({args.calls} micro-steps, lr=0, linear timesteps)...")
    job = get_job(cfg)
    try:
        job.run()
        print("[probe] WARNING: trainer ended before collecting --calls; using what we have")
    except _StopProbe:
        print(f"[probe] collected {len(rows)} samples; building outputs")
    finally:
        try:
            job.cleanup()
        except Exception:
            pass

    if not rows:
        print("[probe] no samples decoded — is the cache built / are faces detected? Aborting.")
        sys.exit(1)

    _write_outputs(rows, sheet, seen, out_dir, args)


def _write_outputs(rows, sheet, seen, out_dir, args):
    arr = np.array([(s, c, l) for (s, c, l, _) in rows], dtype=np.float64)
    sig, cos, lum = arr[:, 0], arr[:, 1], arr[:, 2]
    edges = np.linspace(0, 1, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_cos, mean_lum, counts = [], [], []
    for b in range(args.bins):
        m = (sig >= edges[b]) & (sig < edges[b + 1] if b < args.bins - 1 else sig <= edges[b + 1])
        counts.append(int(m.sum()))
        mean_cos.append(float(cos[m].mean()) if m.any() else float("nan"))
        mean_lum.append(float(lum[m].mean()) if m.any() else float("nan"))

    # --- CSV + console table ---
    csv_path = os.path.join(out_dir, "curve.csv")
    with open(csv_path, "w") as f:
        f.write("sigma_bin_center,mean_cos,mean_lum_std,n\n")
        for c, mc, ml, n in zip(centers, mean_cos, mean_lum, counts):
            f.write(f"{c:.3f},{mc:.4f},{ml:.4f},{n}\n")
    print("\n  sigma   mean_cos   lum_std(shading)   n")
    for c, mc, ml, n in zip(centers, mean_cos, mean_lum, counts):
        print(f"  {c:4.2f}    {mc:6.3f}        {ml:7.2f}        {n}")

    # --- shading-onset heuristic: where lum_std rises off its high-sigma (flat) baseline ---
    valid = [(c, ml) for c, ml in zip(centers, mean_lum) if not np.isnan(ml)]
    hint = None
    if len(valid) >= 3:
        vals = np.array([v for _, v in valid])
        base = float(np.nanmin(vals))               # flattest (high-sigma) shading energy
        rng = float(np.nanmax(vals) - base)
        if rng > 1e-6:
            thr = base + 0.25 * rng                 # 25% up from flat = shading clearly appearing
            # walk from high sigma down; first center (descending) that exceeds thr
            for c, v in sorted(valid, key=lambda x: -x[0]):
                if v > thr:
                    hint = c
                    break

    # --- contact sheet: ROW = one fixed image, COLUMN = sigma bin (same face across a row) ---
    if sheet and seen:
        from PIL import ImageDraw
        cell, label_h = args.cell, 16
        cols, rowsN = args.bins, len(seen)
        canvas = Image.new("RGB", (cols * cell, rowsN * cell + label_h), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for b in range(cols):
            draw.text((b * cell + 4, 3), f"{centers[b]:.2f}", fill=(200, 200, 200))
        for r, path in enumerate(seen):
            for b in range(cols):
                crop = sheet.get((path, b))
                if crop is not None:
                    canvas.paste(Image.fromarray(crop).resize((cell, cell)), (b * cell, label_h + r * cell))
        sheet_path = os.path.join(out_dir, "contact_sheet.png")
        canvas.save(sheet_path)
        print(f"\n[probe] contact sheet -> {sheet_path}")
        print("        ROW = one fixed image; COLUMN = sigma (labelled on top), left=LOW noise (sharp, real")
        print("        shading) -> right=HIGH noise (flat). Read WITHIN a row (same face): the column where")
        print("        that face's shadows fade is its shading cutoff. min_t must be >= that sigma (else the")
        print("        anchor flattens the shaded low-sigma reconstructions). Pick min_t from the consensus.")

    # --- optional curve plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(centers, mean_cos, "o-", color="tab:blue", label="cos to target (identity)")
        ax1.set_xlabel("sigma (t01 = timesteps/1000)"); ax1.set_ylabel("mean cos", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(centers, mean_lum, "s-", color="tab:red", label="luminance std (shading)")
        ax2.set_ylabel("face luminance std (shading energy)", color="tab:red")
        if hint is not None:
            ax1.axvline(hint, color="green", ls="--", label=f"shading onset ~{hint:.2f}")
        fig.legend(loc="upper center", ncol=3, fontsize=8)
        fig.tight_layout()
        plot_path = os.path.join(out_dir, "curve.png")
        fig.savefig(plot_path, dpi=120)
        print(f"[probe] curves -> {plot_path}")
    except Exception as e:
        print(f"[probe] (matplotlib plot skipped: {e})")

    print("\n[probe] READ THE CONTACT SHEET: scan each row left->right; the column where shadows/contour")
    print("        first appear is the shading-onset sigma. Set min_t just ABOVE that column.")
    if hint is not None:
        print(f"[probe] luminance-std heuristic suggests shading onset around sigma ~{hint:.2f} "
              f"-> try min_t ~{min(0.9, hint + 0.05):.2f} (confirm against the sheet).")


if __name__ == "__main__":
    main()
