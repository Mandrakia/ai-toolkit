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


def load_arcface(onnx_path: str, device):
    import onnx2torch
    model = onnx2torch.convert(onnx_path).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model  # fp32; feed via arcface_embed


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
