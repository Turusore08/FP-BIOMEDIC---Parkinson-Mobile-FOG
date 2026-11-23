"""DWPose-m + PatientLock-2Pass extraction worker (one process per GPU).

Mirrors the team YOLO extraction notebook's `_yolo_patientlock_worker.py`, with YOLO26n-pose replaced by
rtmlib YOLOX-m (detector, run inside the patient crop) + DWPose-m (top-down pose on the matched box only).

Run as:  python _dwpose_extract_worker.py job.json
job = {"gpu_label": int,
       "cfg": {"out_dir", "det_url", "pose_url", "max_frames", "device", "det_conf",
               "crop_pad_ratio", "mask_supervisors", "min_crop_iou"},
       "tasks": [{"split", "trial_id", "video_path", "bbox_npz_path"}, ...]}
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from sultan_dwpose.patient_lock import _crop_and_mask_patient, frame_supervisor_boxes, pick_box_in_crop  # inline-strip

DET_INPUT_SIZE = (640, 640)
POSE_INPUT_SIZE = (192, 256)
SELECTOR = "patientlock-2pass"


def build_models(det_url, pose_url, device, det_conf=0.20):
    """det_fn(img) -> (boxes (N,4) xyxy, scores (N,)) with the team's DET_CONF; pose_fn(img, boxes) -> (kps, scores).
    rtmlib's YOLOX hard-codes score > 0.3 for this NMS-in-graph model, so scores are read from the ONNX output."""
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

    def det_fn(img):
        x, ratio = det.preprocess(img)
        out = np.asarray(det.inference(x)[0])
        if out.shape[-1] != 5:
            raise RuntimeError(f"Unexpected YOLOX output shape {out.shape} (expected NMS-in-graph, last dim 5)")
        boxes, scores = out[0, :, :4] / ratio, out[0, :, 4]
        keep = scores > det_conf
        return boxes[keep].astype(np.float32), scores[keep].astype(np.float32)

    def pose_fn(img, boxes):
        return pose(img, bboxes=boxes)

    det_fn.providers = list(det.session.get_providers())
    pose_fn.providers = list(pose.session.get_providers())
    return det_fn, pose_fn


def check_providers(providers, device):
    """onnxruntime silently falls back to CPU when CUDA/cuDNN libs are missing; fail loudly instead."""
    if str(device).startswith("cuda") and "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"Model DWPose jatuh ke CPU (providers={providers}) padahal device={device}. "
                           "Cek instalasi onnxruntime-gpu / CUDA / cuDNN.")


def extract_video(video_path, bbox_npz_path, det_fn, pose_fn, crop_pad_ratio, mask_supervisors, min_crop_iou,
                  max_frames=None):
    bb = np.load(bbox_npz_path)
    pboxes_norm, is_det = bb["patient_box_norm"], bb["is_detected"]
    non_pat_flat, non_pat_off = bb["non_patient_flat"], bb["non_patient_offsets"]
    n_bb = len(pboxes_norm)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.98)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_kps, all_boxes, all_valid, all_n = [], [], [], []
    t0 = time.perf_counter()
    f = 0
    while max_frames is None or f < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        kp, box, valid, n = np.zeros((133, 3), np.float32), np.zeros(4, np.float32), False, 0
        if f < n_bb and bool(is_det[f]):
            sup_b = frame_supervisor_boxes(non_pat_flat, non_pat_off, f)
            crop, off_xy, core_box, pbox_px = _crop_and_mask_patient(
                frame, pboxes_norm[f], sup_b, crop_pad_ratio, mask_supervisors)
            box = pbox_px.astype(np.float32)
            boxes, _scores = det_fn(crop)
            boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
            n = len(boxes)
            idx = pick_box_in_crop(boxes, core_box, min_crop_iou)
            if idx is not None:
                kps, scores = pose_fn(crop, boxes[idx:idx + 1])
                kp = np.concatenate([np.asarray(kps[0], np.float32), np.asarray(scores[0], np.float32)[:, None]], axis=-1)
                kp[:, 0] += float(off_xy[0])
                kp[:, 1] += float(off_xy[1])
                valid = True
        all_kps.append(kp)
        all_boxes.append(box)
        all_valid.append(valid)
        all_n.append(n)
        f += 1
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
        "selector": np.array(SELECTOR),
    }


def is_complete(path):
    """A raw npz counts as done only if it loads, was made with PatientLock (selector marker), came from a
    full run (max_frames=-1), and covers the whole video (1 frame of decoder slack allowed)."""
    try:
        with np.load(path) as z:
            if str(z["selector"]) != SELECTOR or int(z["max_frames"]) != -1:
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
    det_fn, pose_fn = build_models(cfg["det_url"], cfg["pose_url"], device, cfg.get("det_conf", 0.20))
    for fn in (det_fn, pose_fn):
        if hasattr(fn, "providers"):
            check_providers(fn.providers, device)
    for item in pending:
        save_path = out_dir / item["split"] / f"{item['trial_id']}.npz"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        res = extract_video(item["video_path"], item["bbox_npz_path"], det_fn, pose_fn,
                            cfg["crop_pad_ratio"], cfg["mask_supervisors"], cfg["min_crop_iou"],
                            cfg.get("max_frames"))
        save_npz_atomic(save_path, res)
        print(f"[GPU {gpu_label}] Selesai {item['split']}/{item['trial_id']}: {len(res['valid'])} frame | "
              f"valid={res['valid'].mean() * 100:.2f}% | {float(res['ms_per_frame']):.1f} ms/frame", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
