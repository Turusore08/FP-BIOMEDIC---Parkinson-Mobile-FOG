"""Fake rtmlib for the end-to-end notebook dry run (no real models).

YOLOX mimics the NMS-in-graph ONNX model: preprocess -> (img, ratio), inference -> [array (1, N, 5)].
Inside a patient crop it returns the patient (central box) plus a low-score edge box; every 9th call
returns only the edge box (-> no IoU match -> invalid frame), so QC sees some invalid frames.
"""
from types import SimpleNamespace

import numpy as np


class _Base:
    def __init__(self, url, model_input_size=None, backend="onnxruntime", device="cpu", **kw):
        self.session = SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.i = -1


class YOLOX(_Base):
    def preprocess(self, img):
        return img, 1.0

    def inference(self, img):
        self.i += 1
        H, W = img.shape[:2]
        edge = [0, 0, W * 0.08, H * 0.3, 0.25]
        if self.i % 9 == 8:
            return [np.array([[edge]], np.float32)]
        patient = [W * 0.12, H * 0.10, W * 0.88, H * 0.95, 0.9]
        return [np.array([[patient, edge]], np.float32)]

    def __call__(self, img):  # not used by the worker (it reads scores itself)
        out = self.inference(img)[0][0]
        return out[out[:, 4] > 0.3, :4]


class RTMPose(_Base):
    def __call__(self, image, bboxes=()):
        kps, scores = [], []
        for b in bboxes:
            x0, y0, x1, y1 = b
            t = self.i = self.i + 1
            k = np.zeros((133, 2), np.float32)
            k[:, 0] = np.linspace(x0, x1, 133)
            k[:, 1] = np.linspace(y0, y1, 133)
            cx, w, h = 0.5 * (x0 + x1), x1 - x0, y1 - y0
            ph = 0.1 * np.sin(2 * np.pi * t / 30.0)
            for j, (dx, fy) in {11: (-0.15, 0.5), 12: (0.15, 0.5), 13: (-0.15 + ph, 0.7), 14: (0.15 - ph, 0.7),
                                15: (-0.15 + ph, 0.9), 16: (0.15 - ph, 0.9)}.items():
                k[j] = (cx + dx * w, y0 + fy * h)
            kps.append(k)
            scores.append(np.full(133, 0.9, np.float32))
        return np.stack(kps), np.stack(scores)


class Wholebody:
    pass
