import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from sultan_dwpose import extract_worker as ew


def _make_video(path: Path, n=12, w=64, h=48, fps=30.0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        frame = np.full((h, w, 3), (i * 20) % 255, np.uint8)
        vw.write(frame)
    vw.release()
    return path


def _fake_models(empty_frames=()):
    state = {"i": -1}

    def det_fn(frame):
        state["i"] += 1
        if state["i"] in empty_frames:
            return np.zeros((0, 4), np.float32)
        return np.array([[10, 5, 50, 45]], np.float32)

    def pose_fn(frame, boxes):
        n = len(boxes)
        kps = np.tile(np.linspace(0, 40, 133, dtype=np.float32)[None, :, None], (n, 1, 2))
        return kps, np.full((n, 133), 0.8, np.float32)

    return det_fn, pose_fn


def test_extract_shapes_dtypes(tmp_path):
    vp = _make_video(tmp_path / "v.mp4")
    out = ew.extract_video(str(vp), *_fake_models())
    assert out["kps"].shape == (12, 133, 3) and out["kps"].dtype == np.float32
    assert out["boxes"].shape == (12, 4) and out["boxes"].dtype == np.float32
    assert out["valid"].dtype == bool and out["valid"].all()
    assert out["n_persons"].dtype == np.int16 and (out["n_persons"] == 1).all()
    assert int(out["W"]) == 64 and int(out["H"]) == 48
    assert abs(float(out["fps"]) - 30.0) < 0.5
    assert out["ms_per_frame"].dtype == np.float32
    np.testing.assert_allclose(out["kps"][0, :, 2], 0.8)


def test_extract_no_person_frames(tmp_path):
    vp = _make_video(tmp_path / "v.mp4")
    out = ew.extract_video(str(vp), *_fake_models(empty_frames={3, 4, 5}))
    assert not out["valid"][3:6].any() and out["valid"][[0, 1, 2, 6]].all()
    assert not out["kps"][3:6].any()
    assert (out["n_persons"][3:6] == 0).all()


def test_extract_max_frames(tmp_path):
    vp = _make_video(tmp_path / "v.mp4")
    out = ew.extract_video(str(vp), *_fake_models(), max_frames=5)
    assert out["kps"].shape[0] == 5


def _complete_arrays(n=12, max_frames=-1, n_video=12):
    return {"kps": np.zeros((n, 133, 3), np.float32), "valid": np.ones(n, bool),
            "max_frames": np.int32(max_frames), "n_video": np.int32(n_video)}


def test_extract_records_max_frames_and_n_video(tmp_path):
    vp = _make_video(tmp_path / "v.mp4")
    full = ew.extract_video(str(vp), *_fake_models())
    assert int(full["max_frames"]) == -1 and int(full["n_video"]) == 12
    part = ew.extract_video(str(vp), *_fake_models(), max_frames=5)
    assert int(part["max_frames"]) == 5


def test_is_complete_variants(tmp_path):
    p = tmp_path / "t.npz"
    assert ew.is_complete(p) is False                      # missing
    p.write_bytes(b"not a zip")
    assert ew.is_complete(p) is False                      # corrupt
    np.savez(p, a=np.arange(3))
    assert ew.is_complete(p) is False                      # legacy / missing keys
    np.savez(p, **_complete_arrays(n=5, max_frames=5))
    assert ew.is_complete(p) is False                      # test run with MAX_FRAMES
    np.savez(p, **_complete_arrays(n=8, n_video=12))
    assert ew.is_complete(p) is False                      # truncated
    np.savez(p, **_complete_arrays(n=11, n_video=12))
    assert ew.is_complete(p) is True                       # 1-frame decoder slack allowed
    np.savez(p, **_complete_arrays())
    assert ew.is_complete(p) is True


def test_main_reextracts_truncated(tmp_path, monkeypatch):
    vp = _make_video(tmp_path / "v.mp4")
    final = tmp_path / "out" / "val" / "PDFE08_1.npz"
    final.parent.mkdir(parents=True)
    np.savez(final, **_complete_arrays(n=5, max_frames=5))
    monkeypatch.setattr(ew, "build_models", lambda *a, **k: _fake_models())
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "gpu_label": 0,
        "cfg": {"out_dir": str(tmp_path / "out"), "det_url": "d", "pose_url": "p", "max_frames": None},
        "tasks": [{"split": "val", "trial_id": "PDFE08_1", "video_path": str(vp)}],
    }))
    ew.main(str(job))
    assert np.load(final)["kps"].shape[0] == 12 and ew.is_complete(final)


def test_tmp_name_has_pid(tmp_path, monkeypatch):
    seen = []
    real_replace = ew.os.replace
    monkeypatch.setattr(ew.os, "replace", lambda s, d: (seen.append(str(s)), real_replace(s, d)))
    ew.save_npz_atomic(tmp_path / "PDFE02_1.npz", {"a": np.arange(3)})
    assert str(os.getpid()) in Path(seen[0]).name and seen[0].endswith(".tmp.npz")


def test_build_models_preloads_cuda_libs(monkeypatch):
    calls = []

    class FakeTool:
        def __init__(self, *a, **k):
            self.session = types.SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"])

    monkeypatch.setitem(sys.modules, "rtmlib", types.SimpleNamespace(YOLOX=FakeTool, RTMPose=FakeTool))
    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(preload_dlls=lambda: calls.append(1)))
    ew.build_models("d", "p", "cuda")
    assert calls == [1]
    ew.build_models("d", "p", "cpu")
    assert calls == [1]


def test_worker_atomic_and_resume(tmp_path, monkeypatch):
    final = tmp_path / "out" / "train" / "PDFE02_1.npz"
    final.parent.mkdir(parents=True)
    ew.save_npz_atomic(final, _complete_arrays())
    assert final.exists()
    assert not list(final.parent.glob("*.tmp.npz"))
    assert np.load(final)["kps"].shape == (12, 133, 3)

    def boom(*a, **k):
        raise AssertionError("must not run for an already-extracted trial")

    monkeypatch.setattr(ew, "extract_video", boom)
    monkeypatch.setattr(ew, "build_models", boom)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "gpu_label": 0,
        "cfg": {"out_dir": str(tmp_path / "out"), "det_url": "d", "pose_url": "p", "max_frames": None},
        "tasks": [{"split": "train", "trial_id": "PDFE02_1", "video_path": "unused.mp4"}],
    }))
    ew.main(str(job))  # no exception -> skipped


def test_worker_main_extracts_pending(tmp_path, monkeypatch):
    vp = _make_video(tmp_path / "v.mp4")
    monkeypatch.setattr(ew, "build_models", lambda *a, **k: _fake_models())
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "gpu_label": 1,
        "cfg": {"out_dir": str(tmp_path / "out"), "det_url": "d", "pose_url": "p", "max_frames": None},
        "tasks": [{"split": "val", "trial_id": "PDFE08_1", "video_path": str(vp)}],
    }))
    ew.main(str(job))
    z = np.load(tmp_path / "out" / "val" / "PDFE08_1.npz")
    assert set(z.files) >= {"kps", "boxes", "valid", "fps", "W", "H", "n_persons", "ms_per_frame"}
    assert z["kps"].shape == (12, 133, 3)
