from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from dwpose_src.qc_visual import (
    clip_windows,
    pick_sample_trials,
    plot_signals,
    render_clip,
    write_review_template,
)

LB6 = [11, 12, 13, 14, 15, 16]


def _df():
    rows = []
    fogs = [0, 0, 0, 0, 1.0, 2.0, 5.0, 10.0, 25.0, 40.0, 70.0, 90.0, 100.0]
    for i, f in enumerate(fogs):
        rows.append({"trial_id": f"PDFE{i + 2:02d}_1", "split": "train" if i % 3 else "val", "fog_pct": f})
    rows += [{"trial_id": "PDFE01_1", "split": "test", "fog_pct": 96.0},
             {"trial_id": "PDFE06_2", "split": "test", "fog_pct": 0.0}]
    return pd.DataFrame(rows)


def test_pick_sample_excludes_test_and_is_deterministic():
    df = _df()
    a = pick_sample_trials(df, seed=42)
    b = pick_sample_trials(df, seed=42)
    assert a == b
    assert len(a) == 6 and len(set(a)) == 6
    test_ids = set(df.loc[df.split == "test", "trial_id"])
    assert not set(a) & test_ids
    fog = df.set_index("trial_id")["fog_pct"]
    assert sum(fog[t] == 0 for t in a) == 2
    pos = df[(df.split != "test") & (df.fog_pct > 0)].fog_pct
    assert sum(fog[t] >= np.percentile(pos, 200 / 3) for t in a) == 2
    assert sum(0 < fog[t] <= np.median(pos) for t in a) == 2


def test_clip_windows_bounds():
    n, fps = 3600, 30.0
    y = np.zeros(n, int)
    y[900:1200] = 1
    events = {"spike": np.array([3000, 3010, 3020]), "lr_swap": np.array([], int), "person_switch": np.array([], int)}
    starts = clip_windows(n, fps, y, events, seed=0)
    clip = int(15 * fps)
    assert 1 <= len(starts) <= 2
    assert all(0 <= s <= n - clip for s in starts)
    assert any(900 - clip <= s <= 900 for s in starts)
    assert any(s <= 3000 and 3020 < s + clip for s in starts)


def test_clip_windows_short_video():
    starts = clip_windows(100, 30.0, np.zeros(100, int), {"spike": np.array([], int)}, seed=0)
    assert starts == [0]


def _video(path: Path, n=40, w=160, h=120):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), 80, np.uint8))
    vw.release()
    return path


def _pose(n=40):
    kps = np.zeros((n, 133, 3), np.float32)
    kps[:, :23, 0] = np.linspace(20, 140, 23)
    kps[:, :23, 1] = np.linspace(10, 110, 23)
    kps[:, :23, 2] = 0.9
    boxes = np.tile(np.array([10, 5, 150, 115], np.float32), (n, 1))
    return kps, boxes, np.ones(n, bool)


def test_render_clip_writes_frames(tmp_path):
    vp = _video(tmp_path / "v.mp4")
    kps, boxes, valid = _pose()
    y = np.zeros(40, int)
    y[10:20] = 1
    out = tmp_path / "clip.mp4"
    n = render_clip(str(vp), kps, boxes, valid, y, start=5, n=30, out_mp4=out, lb6_idx=LB6)
    assert n == 30
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30


REAL_VIDEO = Path(__file__).resolve().parents[2] / "dataset" / "train" / "videos" / "PDFE02_1.mp4"


@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="needs dataset/ video (H.264, inexact seeking)")
def test_render_clip_is_frame_accurate_on_real_video(tmp_path):
    small = lambda fr: cv2.resize(cv2.cvtColor(fr[40:], cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.int16)
    seq = []
    cap = cv2.VideoCapture(str(REAL_VIDEO))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        seq.append(small(fr))
    cap.release()
    n_total = len(seq)
    kps = np.zeros((n_total, 133, 3), np.float32)
    boxes = np.zeros((n_total, 4), np.float32)
    valid = np.zeros(n_total, bool)  # draw nothing except the top text bar
    offsets = {}
    for s in range(97, n_total - 30, 291):
        out = tmp_path / f"c{s}.mp4"
        render_clip(str(REAL_VIDEO), kps, boxes, valid, np.zeros(n_total, int), s, 1, out, LB6)
        got = small(cv2.VideoCapture(str(out)).read()[1])
        errs = [np.abs(got - seq[s + o]).mean() for o in range(-3, 4)]
        offsets[s] = int(np.argmin(errs)) - 3
    assert all(o == 0 for o in offsets.values()), offsets


def test_plot_signals_creates_png(tmp_path):
    kps, boxes, valid = _pose(300)
    y = np.zeros(300, int)
    y[100:150] = 1
    out = tmp_path / "s.png"
    plot_signals(kps, valid, y, {"spike": np.array([5]), "lr_swap": np.array([7]), "person_switch": np.array([], int)},
                 LB6, "PDFE02_1", out)
    assert out.exists() and out.stat().st_size > 1024


def test_review_template_not_overwritten(tmp_path):
    p = tmp_path / "qc_manual_review.csv"
    assert write_review_template(p, [("PDFE02_1", "PDFE02_1_30s.mp4")]) is True
    first = p.read_text(encoding="utf-8")
    assert first.splitlines()[0] == "trial_id,clip,checked_by,person_ok,left_right_ok,joints_ok,notes"
    assert "PDFE02_1,PDFE02_1_30s.mp4" in first
    assert write_review_template(p, [("PDFE03_1", "x.mp4")]) is False
    assert p.read_text(encoding="utf-8") == first
