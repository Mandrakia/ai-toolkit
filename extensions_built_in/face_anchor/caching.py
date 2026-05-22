"""Per-image / per-dataset metadata the anchor needs at train time.

Built once offline (preflight) because it needs insightface (ONNX detection + ArcFace),
which we deliberately keep OUT of the training venv. The trainer only loads the sidecar.

Cache format (torch.save of a dict):
    {
      "centroids": { dataset_key: np.float32[512] },          # L2-normalized, mean of dataset faces
      "images":    { image_path: {
                        "kps_norm":  np.float32[5,2],          # 5 landmarks, normalized to [0,1] of the image
                        "bbox_norm": np.float32[4],            # x1,y1,x2,y2 normalized
                        "yaw":       float,                     # degrees (for the |yaw|>gate skip)
                        "clean_cos": float,                     # cos(aligned GT emb, centroid) -> pose-fair target
                        "dataset":   dataset_key,
                     } }
    }

kps stored NORMALIZED so they survive aspect-preserving resize into any bucket.
TODO: if the data pipeline center-crops (not just resizes), store the crop transform too
and apply it train-side; current mapping assumes resize-only.
"""
import os
import hashlib
import torch
import numpy as np

# Bump to invalidate every on-disk cache when the detection/embedding pipeline changes.
CACHE_VERSION = 1


class FaceAnchorCache:
    def __init__(self, data: dict):
        self.images = data.get("images", {})
        self._centroids = {k: torch.tensor(np.asarray(v, dtype=np.float32))
                           for k, v in data.get("centroids", {}).items()}
        # realpath index so symlink / abspath differences between preflight and dataloader still match
        self._by_real = {}
        for k, v in self.images.items():
            try:
                self._by_real[os.path.realpath(k)] = v
            except Exception:
                pass

    @classmethod
    def load(cls, path: str) -> "FaceAnchorCache":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"face_anchor cache not found at {path}. Build it with "
                f"`python -m extensions_built_in.face_anchor.caching` (needs insightface).")
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    def get(self, image_path):
        m = self.images.get(image_path)
        if m is None:
            m = self._by_real.get(os.path.realpath(image_path))
        return m

    def centroid(self, dataset_key, device, dtype=torch.float32):
        c = self._centroids.get(dataset_key)
        return None if c is None else c.to(device, dtype)


# --------------------------------------------------------------------------------------
# Preflight builder (run offline, in a venv that has insightface).
# --------------------------------------------------------------------------------------
def build_cache(datasets, out_path, glt_db=None, centroid_override=None,
                model_name="buffalo_l", det_size=640):
    """datasets: list of (dataset_key, image_dir). Multiple dirs can share one key (a group).

    Centroid source priority: centroid_override[key] > GLT folder centroid (glt_db) > computed from
    the dir's single-face images. Mirrors the validated GLT pipeline: insightface detect -> pick
    target face (max cos to centroid) -> aligned 112 emb -> store normalized kps/bbox/yaw + clean_cos.
    landmark_3d_68 is enabled so f.pose (yaw) is populated for the yaw_gate.
    """
    import cv2
    from PIL import Image, ImageOps
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=model_name, allowed_modules=["detection", "recognition", "landmark_3d_68"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(det_size, det_size))

    centroids = _load_glt_centroids(glt_db) if glt_db else {}
    if centroid_override:
        centroids.update(centroid_override)
    images = {}
    for key, image_dir in datasets:
        files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        embeds = []
        # pass 1: provisional centroid if not provided by GLT
        cached = []
        for p in files:
            try:  # EXIF-transpose to match GLT + the ai-toolkit dataloader; insightface wants BGR
                rgb = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
            except Exception:
                continue
            bgr = rgb[:, :, ::-1].copy()
            faces = app.get(bgr)
            if not faces:
                continue
            cached.append((p, bgr.shape[1], bgr.shape[0], faces))
        if key not in centroids:
            for _, _, _, faces in cached:
                if len(faces) == 1:
                    e = faces[0].embedding
                    embeds.append(e / (np.linalg.norm(e) + 1e-8))
            centroids[key] = (np.mean(embeds, 0) / (np.linalg.norm(np.mean(embeds, 0)) + 1e-8)
                              ).astype(np.float32) if embeds else None
        c = centroids[key]
        if c is None:
            print(f"[{key}] no centroid (no single-face images); skipping")
            continue
        # pass 2: pick target face per image, store metadata
        for p, W, H, faces in cached:
            def cos(f):
                e = f.embedding / (np.linalg.norm(f.embedding) + 1e-8)
                return float(np.dot(e, c))
            f = max(faces, key=cos)
            kps = f.kps.astype(np.float32)
            x1, y1, x2, y2 = f.bbox.astype(np.float32)
            yaw = float(f.pose[1]) if getattr(f, "pose", None) is not None else 0.0  # [pitch,yaw,roll]
            images[p] = dict(
                kps_norm=(kps / np.array([W, H], np.float32)),
                bbox_norm=np.array([x1 / W, y1 / H, x2 / W, y2 / H], np.float32),
                yaw=yaw, clean_cos=cos(f), dataset=key,
                embedding=(f.embedding / (np.linalg.norm(f.embedding) + 1e-8)).astype(np.float32),
            )
    torch.save({"centroids": {k: v for k, v in centroids.items() if v is not None}, "images": images}, out_path)
    print(f"wrote {out_path}: {len(images)} images, {len(centroids)} datasets")


def _yaw_from_kps(kps):
    """Geometric yaw proxy (deg) from 5 kps [Leye,Reye,nose,Lmouth,Rmouth] — no pose model needed.
    Frontal -> ~0; nose offset from the eye-midpoint relative to inter-ocular distance -> turn angle."""
    le, re, nose = kps[0], kps[1], kps[2]
    iod = float(np.linalg.norm(re - le)) + 1e-6
    off = (nose[0] - 0.5 * (le[0] + re[0])) / iod
    return float(np.degrees(np.arcsin(np.clip(2.0 * off, -1.0, 1.0))))


DEFAULT_ARCFACE_ONNX = "~/.insightface/models/buffalo_l/w600k_r50.onnx"
DEFAULT_DET_ONNX = "~/.insightface/models/buffalo_l/det_10g.onnx"
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def load_detector_embedder(device="cuda", arcface_onnx=DEFAULT_ARCFACE_ONNX, det_onnx=DEFAULT_DET_ONNX):
    """Load the torch SCRFD detector + onnx2torch ArcFace once, return (scrfd, embed_fn).
    embed_fn(img_rgb, kps) -> L2-normalized float32[512] (same aligned-warp pipeline as training).
    The buffalo_l ONNX models auto-download from HF if not already on disk (no insightface needed)."""
    from .scrfd import SCRFD
    from .geometry import estimate_norm, warp_batch
    from .models import load_arcface, arcface_embed, ensure_onnx_model

    det_path = ensure_onnx_model(det_onnx, "det_10g.onnx")
    scrfd = SCRFD(det_path, device)
    af = load_arcface(arcface_onnx, device)   # resolves/downloads w600k_r50 internally

    def embed(img_rgb, kps):
        M = estimate_norm(kps)
        t = torch.from_numpy(np.ascontiguousarray(img_rgb)).to(device).float().permute(2, 0, 1).unsqueeze(0)
        aligned = warp_batch(t, torch.from_numpy(M).unsqueeze(0).to(device), out=112)
        e = arcface_embed(af, aligned)[0].detach().cpu().numpy()
        return (e / (np.linalg.norm(e) + 1e-8)).astype(np.float32)

    return scrfd, embed


def _decode_rgb(path):
    """PIL decode + EXIF-transpose -> RGB ndarray (matches GLT + the ai-toolkit dataloader), or None."""
    from PIL import Image, ImageOps
    try:
        return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    except Exception:
        return None


def _detect_embed_paths(paths, scrfd, embed, num_workers=8, prefetch=24):
    """Run detection + embedding over an explicit list of image paths.
    Returns list of (path, W, H, faces) where each face dict has 'emb' added. Skips no-face images.

    JPEG/PNG decode is the bottleneck (~43 ms/img CPU vs ~18 ms/img GPU), so decode runs on a thread
    pool (Pillow releases the GIL during decode) with a bounded prefetch window, overlapping it with
    the serial GPU forwards on the main thread -> the loop becomes GPU-bound. RAM ≈ prefetch·imgsize.
    """
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    paths = list(paths)
    out = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        q, idx = deque(), 0
        while idx < len(paths) and len(q) < prefetch:          # prime the prefetch window
            q.append((paths[idx], ex.submit(_decode_rgb, paths[idx]))); idx += 1
        while q:
            p, fut = q.popleft()
            if idx < len(paths):                                # keep the window full
                q.append((paths[idx], ex.submit(_decode_rgb, paths[idx]))); idx += 1
            img = fut.result()
            if img is None:
                continue
            faces = scrfd.detect(img)                           # GPU work stays on the main thread
            if not faces:
                continue
            for f in faces:
                f["emb"] = embed(img, f["kps"])
            out.append((p, img.shape[1], img.shape[0], faces))
    return out


def _two_pass_entries(imgs, key):
    """GLT two-pass over one identity's images (list of (path,W,H,faces[with 'emb'])):
      pass 1: centroid from SINGLE-FACE photos only (multi-face excluded)
      pass 2: keep top-1 face of EVERY photo (max cos to pass-1 centroid), recompute FINAL centroid
    Returns (centroid_f32[512] | None, images_dict)."""
    if not imgs:
        return None, {}
    singles = [faces[0]["emb"] for (_, _, _, faces) in imgs if len(faces) == 1]
    if not singles:
        singles = [max(faces, key=lambda f: f["score"])["emb"] for (_, _, _, faces) in imgs]
    c1 = np.mean(singles, 0); c1 = c1 / (np.linalg.norm(c1) + 1e-8)              # pass-1 centroid
    winners = [(p, W, H, max(faces, key=lambda f: float(np.dot(f["emb"], c1))))
               for (p, W, H, faces) in imgs]                                     # top-1 per photo
    cf = np.mean([f["emb"] for (_, _, _, f) in winners], 0)
    cf = (cf / (np.linalg.norm(cf) + 1e-8)).astype(np.float32)                   # FINAL centroid
    images = {}
    for (p, W, H, f) in winners:
        kps = f["kps"].astype(np.float32)
        x1, y1, x2, y2 = f["bbox"]
        images[p] = dict(
            kps_norm=(kps / np.array([W, H], np.float32)),
            bbox_norm=np.array([x1 / W, y1 / H, x2 / W, y2 / H], np.float32),
            yaw=_yaw_from_kps(kps), clean_cos=float(np.dot(f["emb"], cf)),
            dataset=key, embedding=f["emb"],
        )
    return cf, images


def build_cache_independent(datasets, out_path, device="cuda",
                            arcface_onnx=DEFAULT_ARCFACE_ONNX, det_onnx=DEFAULT_DET_ONNX):
    """Fully self-contained cache build — NO insightface, NO onnxruntime, NO GLT.
    Detection via torch SCRFD (onnx2torch on det_10g), embedding via onnx2torch ArcFace + our own
    aligned warp (same pipeline as training). Several dirs sharing one key = one identity (group),
    centroid computed from the union of single-face images. Validated to reproduce insightface
    (bbox IoU 0.999, embedding cos 0.999).
    """
    scrfd, embed = load_detector_embedder(device, arcface_onnx, det_onnx)
    per_key = {}   # key -> list of (path, W, H, faces[with 'emb'])
    for key, image_dir in datasets:
        paths = [os.path.join(image_dir, fn) for fn in sorted(os.listdir(image_dir))
                 if fn.lower().endswith(_IMAGE_EXTS)]
        per_key.setdefault(key, []).extend(_detect_embed_paths(paths, scrfd, embed))

    centroids, images = {}, {}
    for key, imgs in per_key.items():
        cf, imgs_d = _two_pass_entries(imgs, key)
        if cf is not None:
            centroids[key] = cf
            images.update(imgs_d)
    torch.save({"centroids": centroids, "images": images}, out_path)
    print(f"wrote {out_path}: {len(images)} images, {len(centroids)} datasets (independent / SCRFD)")


# --------------------------------------------------------------------------------------
# In-run cache: per-dataset sidecar + run-level aggregate, both keyed to detect data changes.
# --------------------------------------------------------------------------------------
def _quick_sig(path):
    """size:mtime fingerprint (matches toolkit.basic.get_quick_signature_string) — detects edits."""
    try:
        st = os.stat(path)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except Exception:
        return "missing"


def dataset_signature(paths):
    """Content fingerprint of a dataset's image set: sorted (realpath, size:mtime) + CACHE_VERSION.
    Changes when an image is added, removed, or edited -> triggers a rebuild."""
    h = hashlib.md5(f"v{CACHE_VERSION}".encode())
    for p in sorted(set(os.path.realpath(x) for x in paths)):
        h.update(p.encode()); h.update(b"\0"); h.update(_quick_sig(p).encode()); h.update(b"\n")
    return h.hexdigest()


def aggregate_signature(per_dataset_sigs):
    """Fingerprint of the whole run: sorted (cache_path, dataset_signature) pairs. Changes when any
    dataset's content changes OR the set of datasets changes -> triggers an aggregate rebuild."""
    h = hashlib.md5(f"v{CACHE_VERSION}".encode())
    for path, sig in sorted(per_dataset_sigs):
        h.update(path.encode()); h.update(b"\0"); h.update(sig.encode()); h.update(b"\n")
    return h.hexdigest()


def read_cache_signature(path):
    """Stored signature of an on-disk cache, or None if missing/unreadable/legacy (no signature)."""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False).get("signature")
    except Exception:
        return None


def build_dataset_cache(paths, key, out_path, scrfd, embed, signature):
    """Build one dataset's sidecar cache from its explicit image paths and write it with `signature`."""
    cf, images = _two_pass_entries(_detect_embed_paths(list(paths), scrfd, embed), key)
    centroids = {key: cf} if cf is not None else {}
    torch.save({"centroids": centroids, "images": images,
                "signature": signature, "cache_version": CACHE_VERSION}, out_path)
    print(f"[face_anchor] built {out_path}: {len(images)} faces / {len(paths)} images (key={key})")
    return len(images)


def aggregate_dataset_caches(per_dataset_paths, out_path, global_key, signature):
    """Merge per-dataset sidecars into the run cache: union all face entries, recompute ONE global
    centroid (all datasets in a run = one identity), re-anchor clean_cos + dataset to that centroid.
    per_image targets (each image's own embedding) are preserved untouched."""
    images = {}
    for p in per_dataset_paths:
        if not os.path.exists(p):
            continue
        images.update(torch.load(p, map_location="cpu", weights_only=False).get("images", {}))
    embs = [np.asarray(m["embedding"], np.float32) for m in images.values() if m.get("embedding") is not None]
    if embs:
        c = np.mean(embs, 0); c = (c / (np.linalg.norm(c) + 1e-8)).astype(np.float32)
        for m in images.values():
            m["dataset"] = global_key
            if m.get("embedding") is not None:
                m["clean_cos"] = float(np.dot(np.asarray(m["embedding"], np.float32), c))
        centroids = {global_key: c}
    else:
        centroids = {}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"centroids": centroids, "images": images,
                "signature": signature, "cache_version": CACHE_VERSION}, out_path)
    print(f"[face_anchor] aggregated run cache {out_path}: {len(images)} faces, "
          f"{len(per_dataset_paths)} dataset(s), 1 global centroid (key={global_key})")


def prepare_run_cache(dataset_specs, agg_path, device="cuda",
                      arcface_onnx=DEFAULT_ARCFACE_ONNX, det_onnx=DEFAULT_DET_ONNX,
                      cache_filename="face_anchor.pt", global_key="global"):
    """Orchestrate the in-run cache. `dataset_specs`: list of (key, sidecar_dir, image_paths).

    1. Per dataset: if `<sidecar_dir>/<cache_filename>` is stale/missing (signature mismatch),
       rebuild it (loads SCRFD/ArcFace lazily, once, only if any rebuild is needed).
    2. Aggregate the sidecars into `agg_path` if its signature is stale/missing.
    Returns `agg_path`. Idempotent: a no-change rerun loads zero models and rewrites nothing.
    """
    sidecars, per_sigs, todo = [], [], []
    for key, sidecar_dir, paths in dataset_specs:
        if not paths:
            continue
        out = os.path.join(sidecar_dir, cache_filename)
        sig = dataset_signature(paths)
        sidecars.append(out)
        per_sigs.append((out, sig))
        if read_cache_signature(out) != sig:
            todo.append((key, paths, out, sig))

    if todo:
        scrfd, embed = load_detector_embedder(device, arcface_onnx, det_onnx)
        try:
            for key, paths, out, sig in todo:
                build_dataset_cache(paths, key, out, scrfd, embed, sig)
        finally:
            del scrfd, embed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print(f"[face_anchor] all {len(sidecars)} dataset sidecars up to date")

    agg_sig = aggregate_signature(per_sigs)
    if read_cache_signature(agg_path) != agg_sig:
        aggregate_dataset_caches(sidecars, agg_path, global_key, agg_sig)
    else:
        print(f"[face_anchor] run cache {agg_path} up to date")
    return agg_path


def _load_glt_centroids(glt_db):
    import sqlite3, base64
    con = sqlite3.connect(f"file:{glt_db}?mode=ro", uri=True)
    out = {}
    for sk, cb in con.execute("SELECT scope_key, centroid_b64 FROM centroids WHERE scope_kind='folder'"):
        v = np.frombuffer(base64.b64decode(cb), dtype="<f4").astype(np.float32)
        out[os.path.basename(sk.rstrip("/"))] = v / (np.linalg.norm(v) + 1e-8)
    con.close()
    return out


def load_glt_group_centroid(glt_db, group_key):
    """Group centroid (scope_kind='group') — one identity for all member folders."""
    import sqlite3, base64
    con = sqlite3.connect(f"file:{glt_db}?mode=ro", uri=True)
    row = con.execute(
        "SELECT centroid_b64 FROM centroids WHERE scope_kind='group' AND scope_key=?",
        (str(group_key),)).fetchone()
    con.close()
    if row is None:
        raise ValueError(f"no group centroid in {glt_db} for scope_key={group_key}")
    v = np.frombuffer(base64.b64decode(row[0]), dtype="<f4").astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


if __name__ == "__main__":
    # Build a face-anchor cache from GLT. Run in a venv that HAS insightface (the GridLoraTester one):
    #
    #   GROUP (several folders = ONE identity, uses the GLT group centroid):
    #     PYTHONPATH=/home/mandrakia/ai-toolkit /home/mandrakia/llms/GridLoraTester/.venv/bin/python \
    #       -m extensions_built_in.face_anchor.caching \
    #       --glt-db /home/mandrakia/llms/GridLoraTester/ui/data/glt.db \
    #       --group 1 --key cha_global --out /home/mandrakia/ai-toolkit/.face_anchor_cache_cha.pt \
    #       --dirs /home/mandrakia/ai-toolkit/datasets/charlotte_global \
    #              /home/mandrakia/ai-toolkit/datasets/charlotte_jeune \
    #              /home/mandrakia/ai-toolkit/datasets/chaton_photoshoot \
    #              /home/mandrakia/ai-toolkit/datasets/xxx
    #
    #   FOLDERS (each folder = its own identity; centroid pulled from GLT if --glt-db, else computed):
    #     ... -m ...caching --glt-db <db> --out cache.pt --dirs /datasets/michel /datasets/elijah
    #
    # Find a GLT group id: SELECT id,name,paths_json FROM dataset_groups; in glt.db.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dirs", nargs="+", required=True, help="dataset folder(s)")
    ap.add_argument("--glt-db", default=None, help="path to glt.db (reuse GLT centroids)")
    ap.add_argument("--group", default=None,
                    help="GLT group id: all --dirs share this ONE group centroid (one identity). Needs --glt-db.")
    ap.add_argument("--key", default=None, help="dataset key (group/independent: one identity for all --dirs)")
    ap.add_argument("--independent", action="store_true",
                    help="self-contained: torch SCRFD + onnx2torch, NO insightface/onnxruntime/GLT. "
                         "Runs in the ai-toolkit venv. With --key all dirs share one identity; else each dir is its own.")
    a = ap.parse_args()

    if a.independent:
        if a.key:
            datasets = [(a.key, d) for d in a.dirs]          # all dirs = one identity (union centroid)
        else:
            datasets = [(os.path.basename(d.rstrip("/")), d) for d in a.dirs]  # each dir = its own identity
        build_cache_independent(datasets, a.out)
    elif a.group is not None:
        if not a.glt_db:
            raise SystemExit("--group needs --glt-db (the group centroid lives in glt.db)")
        key = a.key or f"group_{a.group}"
        cent = load_glt_group_centroid(a.glt_db, a.group)
        build_cache([(key, d) for d in a.dirs], a.out, centroid_override={key: cent})
    else:
        # each folder is its own identity (key = folder name); centroid from GLT folder or computed
        datasets = [(os.path.basename(d.rstrip("/")), d) for d in a.dirs]
        build_cache(datasets, a.out, glt_db=a.glt_db)
