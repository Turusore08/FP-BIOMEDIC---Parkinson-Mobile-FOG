import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np

from dwpose_src import extract_worker as ew

W, H = 160, 120
PBOX = np.array([0.30, 0.10, 0.70, 0.90], np.float32)          # patient box (normalised)
CFG = {"crop_pad_ratio": 0.15, "mask_supervisors": True, "min_crop_iou": 0.20}


def _make_video(path: Path, n=12, w=W, h=H, fps=30.0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), (i * 20) % 255, np.uint8))
    vw.release()
    return path


def _make_bbox(path: Path, n=12, undetected=(), sup_every=True):
    pb = np.tile(PBOX, (n, 1))
    det = np.ones(n, bool)
    det[list(undetected)] = False
    flat, offs = [], [0]
    for i in range(n):
        if sup_every:
            flat.append([0.0, 0.0, 0.2, 1.0, 0.9])
        offs.append(len(flat))
    np.savez(path, patient_box_norm=pb, is_detected=det,
             non_patient_flat=np.asarray(flat, np.float32).reshape(-1, 5), non_patient_offsets=np.asarray(offs))
    return path


class FakeModels:
    """det returns a box matching the patient core in crop coords (+ an off-patient box); pose echoes it."""

    def __init__(self, miss_frames=(), only_offpatient=()):
        self.i, self.miss, self.offp = -1, set(miss_frames), set(only_offpatient)
        self.pose_calls, self.det_shapes = [], []

    def det_fn(self, img):
        self.i += 1
        self.det_shapes.append(img.shape)
        h, w = img.shape[:2]
        off = [0, 0, 5, 5]
        if self.i in self.miss:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        if self.i in self.offp:
            return np.array([off], np.float32), np.array([0.9], np.float32)
        core = [0.12 * w, 0.10 * h, 0.88 * w, 0.95 * h]
        return np.array([off, core], np.float32), np.array([0.9, 0.8], np.float32)

    def pose_fn(self, img, boxes):
        self.pose_calls.append(len(boxes))
        b = boxes[0]
        kps = np.zeros((1, 133, 2), np.float32)
        kps[0, :, 0] = b[0] + 1.0
        kps[0, :, 1] = b[1] + 2.0
        return kps, np.full((1, 133), 0.8, np.float32)


def _run(tmp_path, n=12, models=None, undetected=(), max_frames=None, n_bbox=None):
    vp = _make_video(tmp_path / "v.mp4", n=n)
    bp = _make_bbox(tmp_path / "b.npz", n=n if n_bbox is None else n_bbox, undetected=undetected)
    m = models or FakeModels()
    out = ew.extract_video(str(vp), str(bp), m.det_fn, m.pose_fn, max_frames=max_frames, **CFG)
    return out, m


def test_extract_shapes_dtypes_and_marker(tmp_path):
    out, m = _run(tmp_path)
    assert out["kps"].shape == (12, 133, 3) and out["kps"].dtype == np.float32
    assert out["boxes"].shape == (12, 4) and out["valid"].all()
    assert out["n_persons"].dtype == np.int16 and (out["n_persons"] == 2).all()
    assert str(out["selector"]) == "patientlock-2pass"
    assert int(out["W"]) == W and int(out["H"]) == H and abs(float(out["fps"]) - 30.0) < 0.5
    assert int(out["max_frames"]) == -1 and int(out["n_video"]) == 12
    assert m.pose_calls == [1] * 12                         # DWPose only on the matched box


def test_boxes_are_patient_box_pixels(tmp_path):
    out, _ = _run(tmp_path)
    np.testing.assert_allclose(out["boxes"][0], PBOX * [W, H, W, H], atol=1e-4)


def test_keypoints_remapped_to_full_frame(tmp_path):
    out, m = _run(tmp_path)
    px1, py1, px2, py2 = PBOX * [W, H, W, H]
    cx1 = int(max(0, np.floor(px1 - 0.15 * (px2 - px1))))
    cy1 = int(max(0, np.floor(py1 - 0.15 * (py2 - py1))))
    ch, cw = m.det_shapes[0][:2]
    exp_x = cx1 + 0.12 * cw + 1.0
    exp_y = cy1 + 0.10 * ch + 2.0
    np.testing.assert_allclose(out["kps"][0, 0, :2], [exp_x, exp_y], atol=1e-3)
    np.testing.assert_allclose(out["kps"][0, :, 2], 0.8)


def test_crop_is_smaller_than_frame(tmp_path):
    _, m = _run(tmp_path)
    assert m.det_shapes[0][1] < W


def test_invalid_when_no_bbox_no_det_or_no_match(tmp_path):
    # FakeModels counts detector CALLS; frame 2 has no bbox so the detector is not called for it:
    # frame 4 -> call 3 (no detection), frame 6 -> call 5 (only an off-patient box)
    out, m = _run(tmp_path, models=FakeModels(miss_frames={3}, only_offpatient={5}), undetected={2})
    for f in (2, 4, 6):
        assert not out["valid"][f] and not out["kps"][f].any()
    assert out["valid"][[0, 1, 3, 5, 7]].all()
    assert out["n_persons"][2] == 0                         # no crop -> detector never ran
    assert len(m.det_shapes) == 11                          # frame 2 skipped (is_detected False)


def test_frames_beyond_bbox_length_are_invalid(tmp_path):
    out, _ = _run(tmp_path, n=12, n_bbox=10)
    assert out["kps"].shape[0] == 12
    assert not out["valid"][10:].any() and out["valid"][:10].all()


def test_extract_max_frames(tmp_path):
    out, _ = _run(tmp_path, max_frames=5)
    assert out["kps"].shape[0] == 5 and int(out["max_frames"]) == 5


def _complete_arrays(n=12, max_frames=-1, n_video=12, selector="patientlock-2pass"):
    d = {"kps": np.zeros((n, 133, 3), np.float32), "valid": np.ones(n, bool),
         "max_frames": np.int32(max_frames), "n_video": np.int32(n_video)}
    if selector is not None:
        d["selector"] = np.array(selector)
    return d


def test_is_complete_variants(tmp_path):
    p = tmp_path / "t.npz"
    assert ew.is_complete(p) is False                      # missing
    p.write_bytes(b"not a zip")
    assert ew.is_complete(p) is False                      # corrupt
    np.savez(p, a=np.arange(3))
    assert ew.is_complete(p) is False                      # legacy / missing keys
    np.savez(p, **_complete_arrays(selector=None))
    assert ew.is_complete(p) is False                      # first run without PatientLock
    np.savez(p, **_complete_arrays(selector="other"))
    assert ew.is_complete(p) is False
    np.savez(p, **_complete_arrays(n=5, max_frames=5))
    assert ew.is_complete(p) is False                      # MAX_FRAMES test run
    np.savez(p, **_complete_arrays(n=8, n_video=12))
    assert ew.is_complete(p) is False                      # truncated
    np.savez(p, **_complete_arrays(n=11, n_video=12))
    assert ew.is_complete(p) is True                       # 1-frame decoder slack
    np.savez(p, **_complete_arrays())
    assert ew.is_complete(p) is True


def _job(tmp_path, vp, bp, split="val", tid="PDFE08_1"):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "gpu_label": 0,
        "cfg": {"out_dir": str(tmp_path / "out"), "det_url": "d", "pose_url": "p", "max_frames": None,
                "det_conf": 0.20, **CFG},
        "tasks": [{"split": split, "trial_id": tid, "video_path": str(vp), "bbox_npz_path": str(bp)}],
    }))
    return job


def test_main_extracts_and_reextracts_legacy(tmp_path, monkeypatch):
    vp = _make_video(tmp_path / "v.mp4")
    bp = _make_bbox(tmp_path / "b.npz")
    final = tmp_path / "out" / "val" / "PDFE08_1.npz"
    final.parent.mkdir(parents=True)
    np.savez(final, **_complete_arrays(selector=None))     # stale npz from the first (non-PatientLock) run
    seen = {}

    def fake_build(det_url, pose_url, device, det_conf):
        seen["det_conf"] = det_conf
        m = FakeModels()
        return m.det_fn, m.pose_fn

    monkeypatch.setattr(ew, "build_models", fake_build)
    ew.main(str(_job(tmp_path, vp, bp)))
    assert seen["det_conf"] == 0.20
    assert ew.is_complete(final) and np.load(final)["kps"].shape[0] == 12


def test_main_skips_complete(tmp_path, monkeypatch):
    final = tmp_path / "out" / "val" / "PDFE08_1.npz"
    final.parent.mkdir(parents=True)
    ew.save_npz_atomic(final, _complete_arrays())
    assert not list(final.parent.glob("*.tmp.npz"))

    def boom(*a, **k):
        raise AssertionError("must not run for an already-extracted trial")

    monkeypatch.setattr(ew, "extract_video", boom)
    monkeypatch.setattr(ew, "build_models", boom)
    ew.main(str(_job(tmp_path, "unused.mp4", "unused.npz")))


def test_tmp_name_has_pid(tmp_path, monkeypatch):
    seen = []
    real_replace = ew.os.replace
    monkeypatch.setattr(ew.os, "replace", lambda s, d: (seen.append(str(s)), real_replace(s, d)))
    ew.save_npz_atomic(tmp_path / "PDFE02_1.npz", {"a": np.arange(3)})
    assert str(os.getpid()) in Path(seen[0]).name and seen[0].endswith(".tmp.npz")


class _FakeYOLOX:
    """Mimics rtmlib YOLOX with NMS inside the ONNX graph: inference -> [array (1, N, 5)]."""

    def __init__(self, *a, **k):
        self.session = types.SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"])

    def preprocess(self, img):
        return img, 0.5

    def inference(self, img):
        return [np.array([[[10, 10, 50, 50, 0.25], [20, 20, 60, 60, 0.9], [0, 0, 4, 4, 0.1]]], np.float32)]


class _FakePose:
    def __init__(self, *a, **k):
        self.session = types.SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"])

    def __call__(self, img, bboxes):
        return np.zeros((len(bboxes), 133, 2)), np.zeros((len(bboxes), 133))


def test_det_fn_uses_team_threshold_and_scales(monkeypatch):
    monkeypatch.setitem(sys.modules, "rtmlib", types.SimpleNamespace(YOLOX=_FakeYOLOX, RTMPose=_FakePose))
    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(preload_dlls=lambda: None))
    det_fn, pose_fn = ew.build_models("d", "p", "cuda", det_conf=0.20)
    boxes, scores = det_fn(np.zeros((8, 8, 3), np.uint8))
    np.testing.assert_allclose(scores, [0.25, 0.9])        # 0.1 dropped, 0.25 kept (rtmlib would drop < 0.3)
    np.testing.assert_allclose(boxes[1], [40, 40, 120, 120])  # divided by preprocess ratio 0.5
    assert det_fn.providers == ["CUDAExecutionProvider"] and pose_fn.providers == ["CUDAExecutionProvider"]


def test_build_models_preloads_cuda_libs(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "rtmlib", types.SimpleNamespace(YOLOX=_FakeYOLOX, RTMPose=_FakePose))
    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(preload_dlls=lambda: calls.append(1)))
    ew.build_models("d", "p", "cuda")
    assert calls == [1]
    ew.build_models("d", "p", "cpu")
    assert calls == [1]
