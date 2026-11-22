"""Fake rtmlib for an end-to-end notebook dry run (no real models)."""
import numpy as np
from types import SimpleNamespace

class _Base:
    def __init__(self, url, model_input_size=None, backend="onnxruntime", device="cpu", **kw):
        self.session = SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.i = -1

class YOLOX(_Base):
    def __call__(self, frame):
        self.i += 1
        H, W = frame.shape[:2]
        # every 400 frames: 40-frame gap with nobody; otherwise patient + assistant at edge
        if (self.i % 400) < 40:
            return np.array([])
        return np.array([[W*0.4, H*0.1, W*0.6, H*0.95], [0, H*0.05, W*0.25, H*0.98]], np.float32)

class RTMPose(_Base):
    def __call__(self, image, bboxes=[]):
        kps, scores = [], []
        for b in bboxes:
            x0, y0, x1, y1 = b
            t = self.i = self.i + 1
            k = np.zeros((133, 2), np.float32)
            k[:, 0] = np.linspace(x0, x1, 133); k[:, 1] = np.linspace(y0, y1, 133)
            cx, w, h = 0.5*(x0+x1), x1-x0, y1-y0
            ph = 0.1*np.sin(2*np.pi*t/30.0)
            for j, (dx, fy) in {11:(-0.15,0.5),12:(0.15,0.5),13:(-0.15+ph,0.7),14:(0.15-ph,0.7),15:(-0.15+ph,0.9),16:(0.15-ph,0.9)}.items():
                k[j] = (cx + dx*w, y0 + fy*h)
            kps.append(k); scores.append(np.full(133, 0.9, np.float32))
        return np.stack(kps), np.stack(scores)

class Wholebody: pass
