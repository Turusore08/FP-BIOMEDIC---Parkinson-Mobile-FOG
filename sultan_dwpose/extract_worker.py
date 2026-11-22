"""DWPose-m extraction worker (one process per GPU), modelled on Rhama's `_yolo_extract_worker.py`.

Run as:  python _dwpose_extract_worker.py job.json
job = {"gpu_label": int,
       "cfg": {"out_dir": str, "det_url": str, "pose_url": str, "max_frames": int | None},
       "tasks": [{"split": str, "trial_id": str, "video_path": str}, ...]}

Every frame (stride 1) at native resolution. Keypoints are saved in pixel coordinates.
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from sultan_dwpose.select_person import select_main_person  # inline-strip

DET_INPUT_SIZE = (640, 640)
POSE_INPUT_SIZE = (192, 256)


def build_models(det_url, pose_url, device):
    """rtmlib YOLOX-m (HumanArt) detector + DWPose-m pose model (not rtmlib's Wholebody class = RTMW)."""
    if str(device).startswith("cuda"):
        # Same as the notebook's main process: torch loads the pip CUDA/cuDNN libs, preload_dlls
        # lets onnxruntime-gpu find them. Without this a worker subprocess can fall back to CPU.
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    from rtmlib import RTMPose, YOLOX

    det = YOLOX(det_url, model_input_size=DET_INPUT_SIZE, backend="onnxruntime", device=device)
    pose = RTMPose(pose_url, model_input_size=POSE_INPUT_SIZE, to_openpose=False,
                   backend="onnxruntime", device=device)

    def det_fn(frame):
        return det(frame)

    def pose_fn(frame, boxes):
        return pose(frame, bboxes=boxes)

    det_fn.providers = list(det.session.get_providers())
    pose_fn.providers = list(pose.session.get_providers())
    return det_fn, pose_fn


def check_providers(providers, device):
    """onnxruntime silently falls back to CPU when CUDA/cuDNN libs are missing; fail loudly instead."""
    if str(device).startswith("cuda") and "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"Model DWPose jatuh ke CPU (providers={providers}) padahal device={device}. "
                           "Cek instalasi onnxruntime-gpu / CUDA / cuDNN.")


def extract_video(video_path, det_fn, pose_fn, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.98)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_kps, all_boxes, all_valid, all_n = [], [], [], []
    t0 = time.perf_counter()
    while max_frames is None or len(all_kps) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = np.asarray(det_fn(frame), np.float32).reshape(-1, 4)
        if len(boxes):
            # rtmlib RTMPose treats an empty bbox list as "whole image", so only call it with boxes
            kps, scores = pose_fn(frame, boxes)
        else:
            kps, scores = np.zeros((0, 133, 2), np.float32), np.zeros((0, 133), np.float32)
        kp, box, valid, n = select_main_person(boxes, kps, scores, W, H)
        all_kps.append(kp)
        all_boxes.append(box)
        all_valid.append(valid)
        all_n.append(n)
    cap.release()

    n_frames = max(len(all_kps), 1)
    return {
        "kps": np.asarray(all_kps, np.float32).reshape(-1, 133, 3),
        "boxes": np.asarray(all_boxes, np.float32).reshape(-1, 4),
        "valid": np.asarray(all_valid, bool),
        "fps": np.float32(fps),
        "W": np.int32(W),
        "H": np.int32(H),
        "n_persons": np.asarray(all_n, np.int16),
        "ms_per_frame": np.float32(1000.0 * (time.perf_counter() - t0) / n_frames),
        "max_frames": np.int32(-1 if max_frames is None else max_frames),
        "n_video": np.int32(n_video),
    }


def is_complete(path):
    """A raw npz counts as done only if it loads, came from a full run (max_frames=-1),
    and covers the whole video (1 frame of decoder slack allowed)."""
    try:
        with np.load(path) as z:
            if int(z["max_frames"]) != -1:
                return False
            return len(z["valid"]) >= int(z["n_video"]) - 1 and z["kps"].shape[0] == len(z["valid"])
    except Exception:
        return False


def save_npz_atomic(path, arrays):
    path = Path(path)
    # pid in the temp name: two workers (e.g. after a kernel interrupt + re-run) never share a temp file
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def main(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    gpu_label, cfg = job["gpu_label"], job["cfg"]
    out_dir = Path(cfg["out_dir"])
    pending = [t for t in job["tasks"] if not is_complete(out_dir / t["split"] / f"{t['trial_id']}.npz")]
    if not pending:
        print(f"[GPU {gpu_label}] Semua trial sudah diekstraksi, tidak ada yang dikerjakan.", flush=True)
        return

    device = cfg.get("device", "cuda")
    det_fn, pose_fn = build_models(cfg["det_url"], cfg["pose_url"], device)
    for fn in (det_fn, pose_fn):
        if hasattr(fn, "providers"):
            check_providers(fn.providers, device)
    for item in pending:
        save_path = out_dir / item["split"] / f"{item['trial_id']}.npz"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        res = extract_video(item["video_path"], det_fn, pose_fn, cfg.get("max_frames"))
        save_npz_atomic(save_path, res)
        print(f"[GPU {gpu_label}] Selesai {item['split']}/{item['trial_id']}: {len(res['valid'])} frame | "
              f"valid={res['valid'].mean() * 100:.2f}% | {float(res['ms_per_frame']):.1f} ms/frame", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
