"""SCRFD (det_10g) face detector in torch via onnx2torch — no onnxruntime, no insightface.

Port of the validated C# OnnxFaceDetector (Mandrasoft.MediaSync) SCRFD decode:
  - det input: resize (keep ratio) into a 640x640 zero-padded canvas, RGB, (px-127.5)/128, NCHW
  - 3 stride levels [8,16,32], 2 anchors/cell, anchor center = (col*stride, row*stride) (no half-cell)
  - bbox = distance2bbox(center, pred*stride)/ratio ; kps = (center + pred*stride)/ratio
  - score thresh 0.5, NMS (Pascal-VOC +1) thresh 0.4
Outputs are grouped by last-dim (1=score, 4=bbox, 10=kps) and ordered by anchor count, so the
result is robust to the model's raw output ordering.
"""
import numpy as np
import torch
import torch.nn.functional as F

STRIDES = [8, 16, 32]
DET_SIZE = 640
DET_THRESH = 0.5
NMS_THRESH = 0.4


def _nms(boxes, scores, thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1); h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thresh]
    return keep


class SCRFD:
    def __init__(self, onnx_path, device):
        from .models import convert_onnx_cached
        self.model = convert_onnx_cached(onnx_path, device)  # cached GraphModule (eval, frozen)
        self.device = device

    @torch.no_grad()
    def detect(self, img_rgb: np.ndarray):
        """img_rgb: (H,W,3) uint8 RGB. -> list of dict(bbox=[x1,y1,x2,y2], kps=(5,2), score) in image px."""
        H, W = img_rgb.shape[:2]
        ratio = min(DET_SIZE / W, DET_SIZE / H)
        newW, newH = int(W * ratio), int(H * ratio)
        t = torch.from_numpy(np.ascontiguousarray(img_rgb)).to(self.device).float().permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(newH, newW), mode="bilinear", align_corners=False)
        canvas = torch.zeros(1, 3, DET_SIZE, DET_SIZE, device=self.device)
        canvas[:, :, :newH, :newW] = t
        blob = (canvas - 127.5) / 128.0

        outs = self.model(blob)
        outs = [outs] if isinstance(outs, torch.Tensor) else list(outs)
        outs = [o.reshape(-1, o.shape[-1]) for o in outs]
        scores = sorted([o for o in outs if o.shape[-1] == 1], key=lambda o: -o.shape[0])
        bboxes = sorted([o for o in outs if o.shape[-1] == 4], key=lambda o: -o.shape[0])
        kpss = sorted([o for o in outs if o.shape[-1] == 10], key=lambda o: -o.shape[0])

        all_b, all_k, all_s = [], [], []
        for si, stride in enumerate(STRIDES):
            sc = scores[si][:, 0]
            bb = bboxes[si]
            kp = kpss[si]
            N = sc.shape[0]
            gw = DET_SIZE // stride
            apc = max(1, N // (gw * gw))
            idx = torch.arange(N, device=self.device)
            cell = idx // apc
            cx = (cell % gw).float() * stride
            cy = (cell // gw).float() * stride
            m = sc > DET_THRESH
            if m.sum() == 0:
                continue
            cx, cy, sc, bb, kp = cx[m], cy[m], sc[m], bb[m], kp[m]
            x1 = (cx - bb[:, 0] * stride) / ratio
            y1 = (cy - bb[:, 1] * stride) / ratio
            x2 = (cx + bb[:, 2] * stride) / ratio
            y2 = (cy + bb[:, 3] * stride) / ratio
            kx = (cx[:, None] + kp[:, 0::2] * stride) / ratio   # (M,5)
            ky = (cy[:, None] + kp[:, 1::2] * stride) / ratio
            all_b.append(torch.stack([x1, y1, x2, y2], 1))
            all_k.append(torch.stack([kx, ky], -1))             # (M,5,2)
            all_s.append(sc)

        if not all_b:
            return []
        boxes = torch.cat(all_b).cpu().numpy()
        kps = torch.cat(all_k).cpu().numpy()
        scs = torch.cat(all_s).cpu().numpy()
        keep = _nms(boxes, scs, NMS_THRESH)
        return [{"bbox": boxes[i], "kps": kps[i], "score": float(scs[i])} for i in keep]
