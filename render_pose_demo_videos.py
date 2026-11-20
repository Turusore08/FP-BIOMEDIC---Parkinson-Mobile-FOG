"""Render 3 demo videos with YOLO26n-pose 17-keypoint skeleton & LB6 FoG HUD overlay."""

from __future__ import annotations

import csv
from pathlib import Path
import cv2
import numpy as np

REPO = Path("/home/turusore08/Desktop/Semester 5/Biomed/FP/FP-BIOMEDIC---Parkinson-Mobile-FOG")
RAW_POSE = REPO / "output" / "mobilefog_rhama" / "01_extracted_poses_raw" / "yolo26n_rhama"
DATA_TVT = REPO / "Data" / "Train Val Test"
OUT_DIR = REPO / "demo_skeleton_videos"

TARGETS = [
    ("01_test_PDFE16_1_skeleton", "test", "PDFE16_1"),
    ("02_val_PDFE27_2_skeleton", "val", "PDFE27_2"),
    ("03_train_PDFE14_1_skeleton", "train", "PDFE14_1"),
]

# COCO 17 skeleton connections
UPPER_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12),
]
LB6_BONES = [
    (11, 12),          # Pelvis (L_Hip - R_Hip)
    (11, 13), (13, 15),  # Left leg (Hip -> Knee -> Ankle)
    (12, 14), (14, 16),  # Right leg (Hip -> Knee -> Ankle)
]
LB6_JOINTS = {11, 12, 13, 14, 15, 16}


def draw_pose_overlay(
    frame: np.ndarray,
    kps: np.ndarray,
    box: np.ndarray,
    is_valid: bool,
    y_fog: int,
    frame_idx: int,
    total_frames: int,
    fps: float,
    title: str,
    sx: float,
    sy: float,
    conf_thr: float = 0.20,
) -> np.ndarray:
    img = frame.copy()
    H, W = img.shape[:2]

    # Colors (BGR)
    c_upper = (235, 175, 70)          # Sky blue-cyan for upper body
    c_lb6 = (55, 65, 245) if y_fog == 1 else (70, 215, 95)   # Red if FoG else Bright Green
    c_box = (55, 65, 245) if y_fog == 1 else (80, 200, 100)

    if is_valid:
        # Bounding box
        x1, y1, x2, y2 = [int(round(v)) for v in [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]]
        if x2 > x1 and y2 > y1:
            cv2.rectangle(img, (x1, y1), (x2, y2), c_box, 2, cv2.LINE_AA)

        pts = [(int(round(kps[i, 0] * sx)), int(round(kps[i, 1] * sy)), float(kps[i, 2])) for i in range(17)]

        # Draw upper body bones
        for a, b in UPPER_BONES:
            if pts[a][2] >= conf_thr and pts[b][2] >= conf_thr:
                cv2.line(img, pts[a][:2], pts[b][:2], c_upper, 2, cv2.LINE_AA)

        # Draw highlighted LB6 lower-body bones (thicker!)
        for a, b in LB6_BONES:
            if pts[a][2] >= conf_thr and pts[b][2] >= conf_thr:
                cv2.line(img, pts[a][:2], pts[b][:2], c_lb6, 4, cv2.LINE_AA)

        # Draw keypoints
        for idx, (px, py, pc) in enumerate(pts):
            if pc < conf_thr:
                continue
            if idx in LB6_JOINTS:
                cv2.circle(img, (px, py), 7, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), 5, c_lb6, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (px, py), 4, c_upper, -1, cv2.LINE_AA)

        lb6_conf = float(np.mean([kps[i, 2] for i in LB6_JOINTS]))
    else:
        lb6_conf = 0.0

    # Semi-transparent top HUD bar
    overlay = img.copy()
    cv2.rectangle(overlay, (16, 14), (W - 16, 86), (18, 24, 38), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    # Left HUD: Trial info + time
    sec = frame_idx / max(fps, 1.0)
    cv2.putText(
        img,
        f"Mobile-FOG | YOLO26n-Pose (17 KP + LB6) | {title}",
        (30, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 248, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Waktu: {sec:05.1f}s  |  Frame: {frame_idx+1}/{total_frames}  |  LB6 Conf: {lb6_conf*100:4.1f}%",
        (30, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (195, 210, 230),
        1,
        cv2.LINE_AA,
    )

    # Right HUD: Status Pill (NORMAL vs FREEZING OF GAIT)
    pill_txt = "FREEZING OF GAIT (FoG)" if y_fog == 1 else "NORMAL GAIT"
    pill_bg = (40, 45, 215) if y_fog == 1 else (45, 165, 70)
    cv2.rectangle(img, (W - 355, 24), (W - 28, 74), pill_bg, -1)
    cv2.rectangle(img, (W - 355, 24), (W - 28, 74), (255, 255, 255), 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(pill_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
    tx = W - 355 + (327 - tw) // 2
    ty = 24 + (50 + th) // 2 - 2
    cv2.putText(img, pill_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

    return img


def render_one(out_stem: str, split: str, tid: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vp = DATA_TVT / split / "videos" / f"{tid}.mp4"
    lp = DATA_TVT / split / "labels_30hz" / f"{tid}.csv"
    npz_p = RAW_POSE / split / f"{tid}.npz"

    d = np.load(npz_p)
    kps_all = d["kps"]
    boxes_all = d["boxes"]
    valid_all = d["valid"]
    npz_w = int(d["W"]) if "W" in d else 1280
    npz_h = int(d["H"]) if "H" in d else 720

    with open(lp) as f:
        ys = [int(r["y_fog"]) for r in csv.DictReader(f)]

    cap = cv2.VideoCapture(str(vp))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.98)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(kps_all), len(ys))

    sx = W / max(npz_w, 1)
    sy = H / max(npz_h, 1)

    out_mp4 = OUT_DIR / f"{out_stem}.mp4"
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    assert writer.isOpened(), f"Failed to open VideoWriter for {out_mp4}"

    # Pick a FoG frame for visual QA preview
    fog_indices = [i for i in range(n_total) if ys[i] == 1 and valid_all[i]]
    preview_idx = fog_indices[len(fog_indices) // 2] if fog_indices else n_total // 2
    preview_saved = False

    idx = 0
    while idx < n_total:
        ret, frame = cap.read()
        if not ret:
            break
        drawn = draw_pose_overlay(
            frame,
            kps_all[idx],
            boxes_all[idx],
            bool(valid_all[idx]),
            ys[idx],
            idx,
            n_total,
            fps,
            f"{split.upper()} / {tid}",
            sx,
            sy,
        )
        writer.write(drawn)
        if idx == preview_idx:
            cv2.imwrite(str(OUT_DIR / f"{out_stem}_preview.jpg"), drawn)
            preview_saved = True
        idx += 1

    cap.release()
    writer.release()
    size_mb = out_mp4.stat().st_size / (1024 * 1024)
    print(f"Rendered {out_mp4.name}: {idx} frames @ {fps:.2f} fps ({W}x{H}), {size_mb:.1f} MB | preview={preview_saved}")


if __name__ == "__main__":
    for out_stem, split, tid in TARGETS:
        render_one(out_stem, split, tid)
    print("All 3 skeleton demo videos rendered successfully.")
