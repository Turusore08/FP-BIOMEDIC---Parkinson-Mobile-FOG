"""Real DWPose-m on CPU for one short clip. Skipped unless RUN_REAL_MODELS=1 (downloads ~150 MB of models)."""
import os
from pathlib import Path

import numpy as np
import pytest

from sultan_dwpose.build_notebook import DET_URL, POSE_URL
from sultan_dwpose.extract_worker import build_models, check_providers, extract_video
from sultan_dwpose.qc_metrics import event_frames, qc_trial
from sultan_dwpose.qc_visual import render_clip

REPO = Path(__file__).resolve().parents[2]
VIDEO = REPO / "dataset" / "train" / "videos" / "PDFE02_1.mp4"
LB6 = [11, 12, 13, 14, 15, 16]

pytestmark = pytest.mark.skipif(os.environ.get("RUN_REAL_MODELS") != "1" or not VIDEO.exists(),
                                reason="set RUN_REAL_MODELS=1 (needs rtmlib, onnxruntime, dataset/)")


def test_real_dwpose_on_one_video():
    det_fn, pose_fn = build_models(DET_URL, POSE_URL, "cpu")
    check_providers(det_fn.providers, "cpu")
    out = extract_video(str(VIDEO), det_fn, pose_fn, max_frames=60)
    print(f"\nms/frame (CPU) = {float(out['ms_per_frame']):.1f}, valid = {out['valid'].mean():.2f}, "
          f"persons/frame = {out['n_persons'].mean():.2f}")
    assert out["kps"].shape == (60, 133, 3)
    assert out["valid"].mean() > 0.9
    lb6_scores = out["kps"][out["valid"]][:, LB6, 2]
    print("median LB6 score per joint:", np.round(np.median(lb6_scores, axis=0), 3))
    assert np.median(lb6_scores) > 0.3

    r = qc_trial(out["kps"], out["valid"], out["boxes"], out["n_persons"], float(out["fps"]), LB6,
                 n_label30=60, n_video=60)
    print("qc:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    clip = Path(os.environ.get("SMOKE_CLIP", str(REPO / "output" / "smoke_clip.mp4")))
    n = render_clip(str(VIDEO), out["kps"], out["boxes"], out["valid"], np.zeros(60, int), 0, 60, clip, LB6)
    assert n >= 30
    np.savez_compressed(clip.with_suffix(".npz"), **out)
