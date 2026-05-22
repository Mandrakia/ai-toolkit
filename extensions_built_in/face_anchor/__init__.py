# Face identity anchor extension for ai-toolkit.
#
# Adds a differentiable ArcFace identity loss alongside the diffusion loss:
#   x0_pred -> taef2 decode (fixed-size face tile) -> landmark-aligned 112 warp
#   -> ArcFace (onnx2torch, frozen) -> 1 - cos(generated, dataset centroid)
#
# See README.md in this folder for the design rationale and the smoke-test results
# that validated each piece (onnx2torch==insightface, differentiable warp, taef2,
# fixed-window tile sizing from the real face-size distribution).
from toolkit.extension import Extension


class FaceAnchorTrainerExtension(Extension):
    uid = "face_anchor_trainer"
    name = "Face Identity Anchor Trainer"

    @classmethod
    def get_process(cls):
        # imports kept lazy so the heavy deps (onnx2torch, taef2) only load when used
        from .FaceAnchorTrainer import FaceAnchorTrainer
        return FaceAnchorTrainer


AI_TOOLKIT_EXTENSIONS = [FaceAnchorTrainerExtension]
