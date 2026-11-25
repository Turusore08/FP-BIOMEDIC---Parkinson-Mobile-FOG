"""Pose-quality metrics per trial. Report-only: nothing here changes data fed to models.

Works on any raw pose npz in the team's format (kps (T,K,3) pixels, valid, boxes), given the
LB6 joint indices for that estimator, so the same QC can be run on YOLO / MediaPipe output.
"""
import numpy as np

QC_THRESHOLDS = {
    "valid_pct_min": 95.0,
    "lowconf_pct_max": 10.0,
    "spike_pct_max": 2.0,
    "lr_swap_per_min_max": 3.0,
    "person_switch_per_min_max": 1.0,
    "conf_low": 0.2,
    "spike_leg_frac": 0.4,
    "swap_ratio": 0.5,
    "swap_min_leg_frac": 0.2,
    "switch_iou": 0.3,
}

LB6_NAMES = ["L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"]
SWAP_PAIRS = [(2, 3), (4, 5)]  # (L, R) knee and ankle, LB6-local indices


def leg_length_px(kps_lb6, valid):
    """Median over valid frames of mean(L, R) hip->knee->ankle length; NaN if no valid frame."""
    xy = kps_lb6[np.asarray(valid, bool), :, :2]
    if len(xy) == 0:
        return float("nan")
    seg = lambda a, b: np.linalg.norm(xy[:, a] - xy[:, b], axis=-1)
    leg_l = seg(0, 2) + seg(2, 4)
    leg_r = seg(1, 3) + seg(3, 5)
    return float(np.median(0.5 * (leg_l + leg_r)))


def _iou(a, b):
    x0 = np.maximum(a[:, 0], b[:, 0])
    y0 = np.maximum(a[:, 1], b[:, 1])
    x1 = np.minimum(a[:, 2], b[:, 2])
    y1 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: np.clip(z[:, 2] - z[:, 0], 0, None) * np.clip(z[:, 3] - z[:, 1], 0, None)
    union = area(a) + area(b) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _event_masks(kps, valid, boxes, lb6_idx):
    """Boolean masks over frame pairs (t-1, t), t = 1..T-1, plus the valid-pair mask and leg length."""
    valid = np.asarray(valid, bool)
    P = np.asarray(kps, np.float32)[:, lb6_idx, :2]
    leg = leg_length_px(np.asarray(kps)[:, lb6_idx], valid)
    pair_ok = valid[1:] & valid[:-1]
    n_pairs = len(pair_ok)
    if not np.isfinite(leg) or n_pairs == 0:
        empty = np.zeros(n_pairs, bool)
        return {"spike_joint": np.zeros((n_pairs, len(lb6_idx)), bool), "lr_swap": empty,
                "person_switch": empty}, pair_ok, leg

    step = np.linalg.norm(P[1:] - P[:-1], axis=-1)  # (T-1, 6)
    spike_joint = (step > QC_THRESHOLDS["spike_leg_frac"] * leg) & pair_ok[:, None]

    swap = np.zeros(n_pairs, bool)
    swap_counts = []
    for l, r in SWAP_PAIRS:
        d_same = (np.linalg.norm(P[1:, l] - P[:-1, l], axis=-1) + np.linalg.norm(P[1:, r] - P[:-1, r], axis=-1))
        d_cross = (np.linalg.norm(P[1:, l] - P[:-1, r], axis=-1) + np.linalg.norm(P[1:, r] - P[:-1, l], axis=-1))
        ev = (d_cross < QC_THRESHOLDS["swap_ratio"] * d_same) & (d_same > QC_THRESHOLDS["swap_min_leg_frac"] * leg) & pair_ok
        swap_counts.append(ev)
        swap |= ev

    b = np.asarray(boxes, np.float32)
    switch = (_iou(b[1:], b[:-1]) < QC_THRESHOLDS["switch_iou"]) & pair_ok
    return {"spike_joint": spike_joint, "lr_swap": swap, "lr_swap_pairs": swap_counts,
            "person_switch": switch}, pair_ok, leg


def event_frames(kps, valid, boxes, lb6_idx):
    """Frame indices t (the later frame of each (t-1, t) pair) of spikes, L/R swaps, person switches."""
    m, _, _ = _event_masks(kps, valid, boxes, lb6_idx)
    to_frames = lambda mask: np.flatnonzero(mask) + 1
    return {"spike": to_frames(m["spike_joint"].any(axis=1)), "lr_swap": to_frames(m["lr_swap"]),
            "person_switch": to_frames(m["person_switch"])}


def qc_trial(kps, valid, boxes, n_persons, fps, lb6_idx, n_label30, n_video, info_joints=None):
    """info_joints: optional {name: keypoint index} reported as lowconf_pct_<name> (info only, never flagged),
    e.g. DWPose feet 17-22 to judge whether LB10 is usable."""
    th = QC_THRESHOLDS
    valid = np.asarray(valid, bool)
    T = len(valid)
    minutes = max(T / float(fps) / 60.0, 1e-9)
    m, pair_ok, leg = _event_masks(kps, valid, boxes, lb6_idx)

    res = {"n_pose": T, "n_label30": int(n_label30), "n_video": int(n_video),
           "valid_pct": 100.0 * valid.mean() if T else 0.0}
    conf = np.asarray(kps, np.float32)[:, lb6_idx, 2][valid]
    for j, name in enumerate(LB6_NAMES):
        res[f"lowconf_pct_{name}"] = 100.0 * float((conf[:, j] < th["conf_low"]).mean()) if len(conf) else 100.0
    for name, k in (info_joints or {}).items():
        c = np.asarray(kps, np.float32)[valid, k, 2]
        res[f"lowconf_pct_{name}"] = 100.0 * float((c < th["conf_low"]).mean()) if len(c) else 100.0

    n_pairs_ok = int(pair_ok.sum())
    res["spike_pct"] = 100.0 * m["spike_joint"].sum() / (n_pairs_ok * len(lb6_idx)) if n_pairs_ok else 0.0
    per_pair = [ev.sum() for ev in m.get("lr_swap_pairs", [])]
    res["lr_swap_per_min"] = float(max(per_pair)) / minutes if per_pair else 0.0
    res["person_switch_per_min"] = float(m["person_switch"].sum()) / minutes
    res["multi_person_pct"] = (100.0 * float((np.asarray(n_persons) >= 2).mean())
                               if n_persons is not None and T else float("nan"))
    res["leg_px"] = leg

    flags = []
    # labels: 1 row of slack (5 real trials have exactly one label row fewer than video frames)
    if abs(T - int(n_label30)) > 1:
        flags.append("frame_mismatch")
    if T < int(n_video) - 1:
        flags.append("video_mismatch")
    if res["valid_pct"] < th["valid_pct_min"]:
        flags.append("valid")
    if any(res[f"lowconf_pct_{n}"] > th["lowconf_pct_max"] for n in LB6_NAMES):
        flags.append("lowconf")
    if res["spike_pct"] > th["spike_pct_max"]:
        flags.append("spike")
    if res["lr_swap_per_min"] > th["lr_swap_per_min_max"]:
        flags.append("lr_swap")
    if res["person_switch_per_min"] > th["person_switch_per_min_max"]:
        flags.append("person_switch")
    res["flags"] = ";".join(flags)
    return res
