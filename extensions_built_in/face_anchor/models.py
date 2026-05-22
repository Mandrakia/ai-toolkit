"""Frozen perceptual models for the anchor: differentiable ArcFace + tiny VAE decoder.

Validated (2026-05-21):
  - onnx2torch(w600k_r50.onnx) == insightface ONNX, cos 1.0; ~18ms/0.6GB bs8 fwd+bwd
  - taef2 (madebyollin/taef2) loads into diffusers AutoencoderTiny(latent_channels=32):
    decoder loads exact (missing=0); only the flux2 encoder pool block is absent (unused here).
    Decoder input [0,1]-space latent API, output ~[0,1]. ~46ms/1.6GB at 1024 full-frame.
"""
import os
import torch
import torch.nn.functional as F

TAEF2_REPO = "madebyollin/taef2"
TAEF2_FILE = "taef2.safetensors"

# buffalo_l ONNX (SCRFD det_10g + ArcFace w600k_r50). Byte-identical to insightface's, but fetched
# WITHOUT insightface/onnxruntime: from this HF mirror if not already on disk under ~/.insightface.
BUFFALO_HF_REPO = "public-data/insightface"


def _convert_cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    d = os.path.join(base, "face_anchor", "onnx2torch")
    os.makedirs(d, exist_ok=True)
    return d


def ensure_onnx_model(path, hf_filename):
    """Local path to a buffalo_l onnx file. Use `path` if it exists (e.g. populated by insightface);
    otherwise download the byte-identical file from HF (no insightface / onnxruntime) into the HF
    cache and return that. Raises with a clear hint if both the local path and the download fail."""
    local = os.path.expanduser(path)
    if os.path.exists(local):
        return local
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(BUFFALO_HF_REPO, f"models/buffalo_l/{hf_filename}")
        print(f"[face_anchor] {hf_filename} not at {local}; fetched from HF {BUFFALO_HF_REPO} (cached)")
        return p
    except Exception as e:
        raise FileNotFoundError(
            f"face_anchor needs {hf_filename} (buffalo_l) and it is not at {local}; the HF fallback "
            f"download failed: {e}. Place the buffalo_l ONNX models under ~/.insightface/models/buffalo_l/ "
            f"or point det_onnx/arcface_onnx at them.")


def _onnx2torch_version():
    try:
        from importlib.metadata import version
        return version("onnx2torch")
    except Exception:
        import onnx2torch
        return getattr(onnx2torch, "__version__", "x")


def convert_onnx_cached(onnx_path: str, device):
    """onnx2torch.convert with a disk cache of the converted GraphModule.

    onnx2torch.convert reparses the ONNX graph every call (~1.9s for r50 ArcFace, ~1.3s for SCRFD);
    torch.load of the pickled module is ~40x faster and bit-identical (validated 2026-05-22, grad
    still flows). Cache key = onnx size:mtime + torch + onnx2torch versions, so a model/lib change
    invalidates it. Any miss/load failure falls back to a fresh convert (and rewrites the cache).
    Returns the module on `device`, eval, params frozen (input grad still flows for the train loss).
    """
    import onnx2torch
    onnx_path = os.path.expanduser(onnx_path)
    try:
        st = os.stat(onnx_path); sig = f"{st.st_size}_{int(st.st_mtime)}"
    except Exception:
        sig = "nosig"
    tag = f"{os.path.basename(onnx_path)}.o2t-t{torch.__version__}-o{_onnx2torch_version()}-{sig}.pt"
    tag = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in tag)
    cache = os.path.join(_convert_cache_dir(), tag)   # dedicated dir (onnx may live in the read-mostly HF cache)

    model = None
    if os.path.exists(cache):
        try:
            model = torch.load(cache, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[face_anchor] converted-model cache unreadable ({os.path.basename(cache)}): {e}; reconverting")
    if model is None:
        model = onnx2torch.convert(onnx_path)
        try:
            torch.save(model, cache)
        except Exception as e:
            print(f"[face_anchor] could not cache converted model to {cache}: {e}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_arcface(onnx_path: str, device):
    # resolve (or HF-download) then convert-with-cache; fp32, feed via arcface_embed
    return convert_onnx_cached(ensure_onnx_model(onnx_path, "w600k_r50.onnx"), device)


def arcface_embed(model, img255: torch.Tensor) -> torch.Tensor:
    """img255: (n,3,112,112) RGB in [0,255]. -> L2-normalized (n,512) embeddings (grad flows in)."""
    x = (img255 - 127.5) / 127.5
    return F.normalize(model(x.float()).float(), dim=-1)


def load_taef2(path, device, dtype):
    """Idempotent: use `path` if set and present, else download madebyollin/taef2 into the HF
    cache (downloads once, reused on every later run). Loads into diffusers AutoencoderTiny(32).
    """
    from diffusers import AutoencoderTiny
    from safetensors.torch import load_file
    if path and os.path.exists(os.path.expanduser(path)):
        weights = os.path.expanduser(path)
    else:
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download(TAEF2_REPO, TAEF2_FILE)   # cached -> idempotent
    ae = AutoencoderTiny(latent_channels=32)
    # decoder loads exact; encoder flux2-pool keys are unexpected (we only use the decoder)
    ae.load_state_dict(load_file(weights), strict=False)
    ae = ae.to(device, dtype).eval()
    ae.requires_grad_(False)
    return ae


def taef2_decode(taef2, latent: torch.Tensor) -> torch.Tensor:
    """latent (n,32,b,b) in the Flux-2 VAE space -> RGB image (n,3,b*8,b*8) in [0,255], grad flows.

    Cast the latent to the decoder's dtype (the fp8-quantized transformer can hand us a float32
    x0_pred while taef2 is bf16). `.to()` is differentiable so the gradient still flows.
    """
    pdtype = next(taef2.parameters()).dtype
    out = taef2.decoder(latent.to(pdtype))   # ~[0,1]
    return out.clamp(0, 1).float() * 255.0
