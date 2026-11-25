"""PatientLock-2Pass helpers for the DWPose worker.

`_box_iou_one_to_many` and `_crop_and_mask_patient` are VERBATIM copies of the team YOLO extraction
notebook's worker (parkinsonfp-biomed.ipynb, cell 4); a test enforces it. Do not edit them here.
"""
import numpy as np


def _box_iou_one_to_many(target_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(target_box[0], boxes[:, 0])
    iy1 = np.maximum(target_box[1], boxes[:, 1])
    ix2 = np.minimum(target_box[2], boxes[:, 2])
    iy2 = np.minimum(target_box[3], boxes[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_t = max(0.0, (target_box[2] - target_box[0]) * (target_box[3] - target_box[1]))
    area_b = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    return inter / np.maximum(area_t + area_b - inter, 1e-6)


def _crop_and_mask_patient(
    frame: np.ndarray,
    pbox_norm: np.ndarray,
    sup_boxes_norm: np.ndarray,
    pad_ratio: float,
    mask_supervisors: bool,
):
    H, W = frame.shape[:2]
    px1 = float(np.clip(pbox_norm[0] * W, 0, W - 1))
    py1 = float(np.clip(pbox_norm[1] * H, 0, H - 1))
    px2 = float(np.clip(pbox_norm[2] * W, px1 + 1, W))
    py2 = float(np.clip(pbox_norm[3] * H, py1 + 1, H))

    bw = px2 - px1
    bh = py2 - py1
    cx1 = int(max(0, np.floor(px1 - pad_ratio * bw)))
    cy1 = int(max(0, np.floor(py1 - pad_ratio * bh)))
    cx2 = int(min(W, np.ceil(px2 + pad_ratio * bw)))
    cy2 = int(min(H, np.ceil(py2 + pad_ratio * bh)))

    crop = frame[cy1:cy2, cx1:cx2].copy()
    ch, cw = crop.shape[:2]
    core_in_crop = np.array([px1 - cx1, py1 - cy1, px2 - cx1, py2 - cy1], dtype=np.float32)

    if mask_supervisors and len(sup_boxes_norm) > 0:
        # Simpan area inti pasien dulu agar tidak ikut tertutup jika bbox pengawas sedikit bersinggungan
        ix1, iy1 = int(max(0, np.floor(core_in_crop[0]))), int(max(0, np.floor(core_in_crop[1])))
        ix2, iy2 = int(min(cw, np.ceil(core_in_crop[2]))), int(min(ch, np.ceil(core_in_crop[3])))
        patient_patch = crop[iy1:iy2, ix1:ix2].copy()

        for sb in sup_boxes_norm:
            sx1 = int(np.clip(np.floor(float(sb[0]) * W) - cx1, 0, cw))
            sy1 = int(np.clip(np.floor(float(sb[1]) * H) - cy1, 0, ch))
            sx2 = int(np.clip(np.ceil(float(sb[2]) * W) - cx1, 0, cw))
            sy2 = int(np.clip(np.ceil(float(sb[3]) * H) - cy1, 0, ch))
            if sx2 > sx1 and sy2 > sy1:
                crop[sy1:sy2, sx1:sx2] = (114, 114, 114)

        crop[iy1:iy2, ix1:ix2] = patient_patch

    pbox_px = np.array([px1, py1, px2, py2], dtype=np.float32)
    return crop, (cx1, cy1), core_in_crop, pbox_px


def pick_box_in_crop(boxes, core_in_crop, min_iou):
    """Index of the detection (crop coords) with the largest IoU vs the patient core box, or None
    if there is none / it is below min_iou (team rule from `_pick_pose_in_crop`)."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return None
    ious = _box_iou_one_to_many(np.asarray(core_in_crop, np.float32), boxes)
    idx = int(np.argmax(ious))
    return idx if ious[idx] >= min_iou else None


def frame_supervisor_boxes(non_patient_flat, non_patient_offsets, f):
    """Supervisor boxes (normalised xyxy) of frame f from the CSR layout of the bbox dataset."""
    s0, s1 = int(non_patient_offsets[f]), int(non_patient_offsets[f + 1])
    if s1 <= s0:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(non_patient_flat[s0:s1, :4], dtype=np.float32)


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
