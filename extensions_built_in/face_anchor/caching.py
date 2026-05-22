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
import torch
import numpy as np


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
            bgr = cv2.imread(p)
            if bgr is None:
                continue
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
    ap.add_argument("--key", default=None, help="dataset key for --group mode (default group_<id>)")
    a = ap.parse_args()

    if a.group is not None:
        if not a.glt_db:
            raise SystemExit("--group needs --glt-db (the group centroid lives in glt.db)")
        key = a.key or f"group_{a.group}"
        cent = load_glt_group_centroid(a.glt_db, a.group)
        build_cache([(key, d) for d in a.dirs], a.out, centroid_override={key: cent})
    else:
        # each folder is its own identity (key = folder name); centroid from GLT folder or computed
        datasets = [(os.path.basename(d.rstrip("/")), d) for d in a.dirs]
        build_cache(datasets, a.out, glt_db=a.glt_db)
