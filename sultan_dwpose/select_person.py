"""Per-frame patient selection, mirroring `_select_main_person` of the team's YOLO notebook.

YOLO notebook: score = box_area * det_conf * (1 - 0.3*|cx - W/2| / (W/2)), argmax, no tracking.
rtmlib's YOLOX returns no detection score, so det_conf is replaced by the mean score
of the 17 COCO body keypoints (indices 0..16) of each person.
"""
import numpy as np

NUM_KPS = 133
BODY_SLICE = slice(0, 17)


def select_main_person(boxes, kps, scores, W, H):
    """boxes (N,4) xyxy, kps (N,133,2), scores (N,133) -> (kp (133,3), box (4,), valid, n_persons)."""
    n = int(len(boxes))
    if n == 0:
        return np.zeros((NUM_KPS, 3), np.float32), np.zeros(4, np.float32), False, 0
    if n == 1:
        idx = 0
    else:
        b = np.asarray(boxes, np.float32)
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        conf = np.asarray(scores, np.float32)[:, BODY_SLICE].mean(axis=1)
        cx = 0.5 * (b[:, 0] + b[:, 2])
        center_penalty = 1.0 - 0.3 * np.abs(cx - W * 0.5) / max(W * 0.5, 1.0)
        idx = int(np.argmax(areas * conf * center_penalty))
    kp = np.concatenate([kps[idx], np.asarray(scores[idx])[:, None]], axis=-1).astype(np.float32)
    return kp, np.asarray(boxes[idx], np.float32), True, n
