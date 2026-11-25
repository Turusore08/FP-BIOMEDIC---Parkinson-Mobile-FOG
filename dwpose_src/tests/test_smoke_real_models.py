"""Real YOLOX + DWPose-m (CPU) with PatientLock on PDFE02_1. Skipped unless RUN_REAL_MODELS=1 (~150 MB models).

The real bbox dataset (ajikajadeh/mobile-fog-patient-bboxes) is not needed: a fixed patient box measured on
this video (patient ~x 452-740 px, right supervisor ~x 762-1048 px) stands in for it. Frames 0-180 include
f129-f145 where the old per-frame rule jumped to the right supervisor.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from dwpose_src.build_notebook import DET_URL, POSE_URL
from dwpose_src.extract_worker import build_models, check_providers, extract_video
from dwpose_src.qc_metrics import qc_trial
from dwpose_src.qc_visual import render_clip

REPO = Path(__file__).resolve().parents[2]
VIDEO = REPO / "dataset" / "train" / "videos" / "PDFE02_1.mp4"
LB6 = [11, 12, 13, 14, 15, 16]
N = 180

pytestmark = pytest.mark.skipif(os.environ.get("RUN_REAL_MODELS") != "1" or not VIDEO.exists(),
                                reason="set RUN_REAL_MODELS=1 (needs rtmlib, onnxruntime, dataset/)")


def _bbox_file(path):
    pb = np.tile(np.array([452 / 1280, 0.0, 740 / 1280, 650 / 720], np.float32), (N, 1))
    sup = np.array([[762 / 1280, 0.0, 1048 / 1280, 560 / 720, 0.9], [150 / 1280, 0.0, 420 / 1280, 560 / 720, 0.9]],
                   np.float32)
    np.savez(path, patient_box_norm=pb, is_detected=np.ones(N, bool),
             non_patient_flat=np.tile(sup, (N, 1)), non_patient_offsets=np.arange(0, 2 * N + 1, 2))
    return path


def test_real_dwpose_patientlock_on_pdfe02(tmp_path):
    det_fn, pose_fn = build_models(DET_URL, POSE_URL, "cpu", det_conf=0.20)
    check_providers(det_fn.providers, "cpu")
    out = extract_video(str(VIDEO), str(_bbox_file(tmp_path / "bb.npz")), det_fn, pose_fn,
                        crop_pad_ratio=0.15, mask_supervisors=True, min_crop_iou=0.20, max_frames=N)
    print(f"\nms/frame (CPU) = {float(out['ms_per_frame']):.1f}, valid = {out['valid'].mean():.3f}, "
          f"dets in crop/frame = {out['n_persons'].mean():.2f}")
    assert out["kps"].shape == (N, 133, 3) and str(out["selector"]) == "patientlock-2pass"
    assert out["valid"].mean() > 0.95
    ankles_x = out["kps"][out["valid"]][:, [15, 16], 0]
    print("ankle x range:", ankles_x.min().round(1), ankles_x.max().round(1))
    assert ankles_x.max() < 762 and ankles_x.min() > 400      # never on the right/left supervisor
    lb6_scores = out["kps"][out["valid"]][:, LB6, 2]
    print("median LB6 score per joint:", np.round(np.median(lb6_scores, axis=0), 3))
    assert np.median(lb6_scores) > 0.3

    r = qc_trial(out["kps"], out["valid"], out["boxes"], out["n_persons"], float(out["fps"]), LB6,
                 n_label30=N, n_video=N)
    print("qc:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    assert r["person_switch_per_min"] == 0.0
    clip = Path(os.environ.get("SMOKE_CLIP", str(REPO / "output" / "smoke_clip.mp4")))
    n = render_clip(str(VIDEO), out["kps"], out["boxes"], out["valid"], np.zeros(N, int), 120, 30, clip, LB6)
    assert n == 30
