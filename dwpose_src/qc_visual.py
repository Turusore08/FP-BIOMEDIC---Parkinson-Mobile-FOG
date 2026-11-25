"""QC visuals: signal plots, 15 s overlay clips for manual checking, sample selection, review sheet."""
import csv
from pathlib import Path

import cv2
import numpy as np

from dwpose_src.qc_metrics import leg_length_px  # inline-strip

CLIP_SEC = 15.0
EVENT_KEYS = ("spike", "lr_swap", "person_switch")

# COCO-WholeBody body (0-16) + feet (17-22) bones
BODY_BONES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
LB6_BONES = {(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)}
LEFT = {1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19}
RIGHT = {2, 4, 6, 8, 10, 12, 14, 16, 20, 21, 22}
COL_L, COL_R, COL_C = (255, 160, 0), (0, 80, 255), (200, 200, 200)  # BGR: left blue, right red
CONF_DRAW = 0.2


def pick_sample_trials(df_fog, seed=42):
    """6 train/val trials: 2 from the top FoG tertile, 2 with 0 < FoG <= median, 2 with FoG == 0."""
    d = df_fog[df_fog["split"].isin(["train", "val"])].sort_values("trial_id")
    pos = d[d["fog_pct"] > 0]["fog_pct"]
    groups = [
        d[d["fog_pct"] >= np.percentile(pos, 200 / 3)] if len(pos) else d.iloc[:0],
        d[(d["fog_pct"] > 0) & (d["fog_pct"] <= np.median(pos))] if len(pos) else d.iloc[:0],
        d[d["fog_pct"] == 0],
    ]
    rng = np.random.default_rng(seed)
    out = []
    for g in groups:
        ids = [t for t in g["trial_id"].tolist() if t not in out]
        out += rng.choice(ids, size=min(2, len(ids)), replace=False).tolist() if ids else []
    return out


def clip_windows(n_frames, fps, y_fog30, events, seed):
    """<= 2 clip start frames: middle of the first FoG episode, and the densest QC-event region."""
    clip = int(round(CLIP_SEC * fps))
    if n_frames <= clip:
        return [0]
    last = n_frames - clip
    clamp = lambda s: int(min(max(s, 0), last))
    starts = []
    y = np.asarray(y_fog30[:n_frames]) > 0
    if y.any():
        s = int(np.argmax(y))
        e = s + int(np.argmin(y[s:])) if (~y[s:]).any() else n_frames
        starts.append(clamp((s + e) // 2 - clip // 2))
    ev = np.concatenate([np.asarray(events.get(k, []), int) for k in EVENT_KEYS]) if events else np.array([], int)
    if len(ev):
        ev = np.sort(ev)
        # centre the clip on the event with the most other events within +-clip/2
        lo = np.searchsorted(ev, ev - clip // 2, side="left")
        hi = np.searchsorted(ev, ev + clip // 2, side="left")
        cand = clamp(int(ev[int(np.argmax(hi - lo))]) - clip // 2)
    else:
        cand = int(np.random.default_rng(seed).integers(0, last + 1))
    if not starts or abs(cand - starts[0]) >= clip // 2:
        starts.append(cand)
    return starts


def plot_signals(kps, valid, y_fog30, events, lb6_idx, title, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = np.asarray(valid, bool)
    P = np.asarray(kps, np.float32)[:, lb6_idx]
    leg = leg_length_px(P, valid)
    scale = leg if np.isfinite(leg) and leg > 0 else 1.0
    xy = np.where(valid[:, None, None], P[..., :2], np.nan) / scale
    t = np.arange(len(P))
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True, dpi=100)
    y = np.asarray(y_fog30)
    for ax, c, lab in zip(axes, (0, 1), ("x / leg", "y / leg")):
        ax.plot(t, xy[:, 4, c], color="tab:blue", lw=0.8, label="L_Ankle")
        ax.plot(t, xy[:, 5, c], color="tab:red", lw=0.8, label="R_Ankle")
        n = min(len(y), len(t))
        ax.fill_between(t[:n], 0, 1, where=y[:n] > 0, transform=ax.get_xaxis_transform(),
                        color="orange", alpha=0.25, label="FoG")
        for k, col in zip(EVENT_KEYS, ("black", "purple", "green")):
            fr = np.asarray(events.get(k, []), int)
            if len(fr):
                ax.vlines(fr, 0, 0.06, transform=ax.get_xaxis_transform(), color=col, lw=0.8, label=k)
        ax.set_ylabel(lab)
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", fontsize=7, ncol=6)
    axes[1].set_xlabel("frame (30 fps)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _joint_color(j):
    return COL_L if j in LEFT else COL_R if j in RIGHT else COL_C


def render_clip(video_path, kps, boxes, valid, y_fog30, start, n, out_mp4, lb6_idx):
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.98)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # read forward instead of CAP_PROP_POS_FRAMES: seeking in these H.264 videos often lands 1 frame early,
    # which would draw the skeleton on the wrong image
    for _ in range(int(start)):
        if not cap.grab():
            break
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    lb6 = set(lb6_idx)
    written = 0
    for i in range(int(start), int(start) + int(n)):
        ok, img = cap.read()
        if not ok or i >= len(kps):
            break
        k = kps[i]
        if valid[i]:
            for a, b in BODY_BONES:
                if k[a, 2] >= CONF_DRAW and k[b, 2] >= CONF_DRAW:
                    thick = 4 if (a, b) in LB6_BONES else 1
                    cv2.line(img, tuple(map(int, k[a, :2])), tuple(map(int, k[b, :2])), _joint_color(b), thick, cv2.LINE_AA)
            for j in range(23):
                if k[j, 2] >= CONF_DRAW:
                    cv2.circle(img, tuple(map(int, k[j, :2])), 6 if j in lb6 else 3, _joint_color(j), -1, cv2.LINE_AA)
            x0, y0, x1, y1 = map(int, boxes[i])
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 1)
        fog = int(y_fog30[i]) if i < len(y_fog30) else -1
        status = {1: "FoG", 0: "NORMAL", -1: "NO LABEL"}[fog]
        score = float(k[list(lb6_idx), 2].mean()) if valid[i] else 0.0
        txt = f"f={i} t={i / fps:6.2f}s  {status}  LB6 score={score:.2f}  {'VALID' if valid[i] else 'NO PERSON'}"
        cv2.rectangle(img, (0, 0), (W, 28), (0, 0, 0), -1)
        cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (60, 60, 255) if fog == 1 else (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
        written += 1
    cap.release()
    vw.release()
    return written


def write_review_template(path, rows):
    path = Path(path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "clip", "checked_by", "person_ok", "left_right_ok", "joints_ok", "notes"])
        for trial_id, clip in rows:
            w.writerow([trial_id, clip, "", "", "", "", ""])
    return True
