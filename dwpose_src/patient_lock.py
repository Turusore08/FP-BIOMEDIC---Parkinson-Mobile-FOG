"""PatientLock-2Pass helpers for the DWPose worker.

`_box_iou_one_to_many` and `_crop_and_mask_patient` are VERBATIM copies of the team YOLO extraction
notebook's worker (01_extract_yolo26_patientlock.ipynb, cell 4); a test enforces it. Do not edit them here.
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
