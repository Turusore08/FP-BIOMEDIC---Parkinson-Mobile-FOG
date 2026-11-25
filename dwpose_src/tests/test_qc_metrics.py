import numpy as np

from dwpose_src.qc_metrics import QC_THRESHOLDS, event_frames, leg_length_px, qc_trial

FPS = 30.0
T = 1800  # 60 s
LB6 = [11, 12, 13, 14, 15, 16]
BASE = {11: (600, 300), 12: (680, 300), 13: (590, 400), 14: (690, 400), 15: (585, 500), 16: (695, 500)}


def make_trial():
    t = np.arange(T) / FPS
    kps = np.zeros((T, 133, 3), np.float32)
    kps[..., 2] = 0.9
    for j, (x, y) in BASE.items():
        kps[:, j, 0] = x + 15 * np.sin(2 * np.pi * t)
        kps[:, j, 1] = y + 5 * np.cos(2 * np.pi * t)
    boxes = np.tile(np.array([560, 250, 740, 560], np.float32), (T, 1))
    valid = np.ones(T, bool)
    n_persons = np.ones(T, np.int16)
    return kps, valid, boxes, n_persons


def run(kps, valid, boxes, n_persons, n_label30=T, n_video=T):
    return qc_trial(kps, valid, boxes, n_persons, FPS, LB6, n_label30=n_label30, n_video=n_video)


def test_thresholds_match_spec():
    assert QC_THRESHOLDS == {
        "valid_pct_min": 95.0, "lowconf_pct_max": 10.0, "spike_pct_max": 2.0,
        "lr_swap_per_min_max": 3.0, "person_switch_per_min_max": 1.0,
        "conf_low": 0.2, "spike_leg_frac": 0.4, "swap_ratio": 0.5, "swap_min_leg_frac": 0.2,
        "switch_iou": 0.3,
    }


def test_leg_length():
    kps, valid, _, _ = make_trial()
    assert abs(leg_length_px(kps[:, LB6], valid) - 200.6) < 1.0
    assert np.isnan(leg_length_px(kps[:, LB6], np.zeros(T, bool)))


def test_qc_clean_trial():
    r = run(*make_trial())
    assert r["flags"] == ""
    assert r["valid_pct"] == 100.0
    assert r["spike_pct"] == 0.0 and r["lr_swap_per_min"] == 0.0 and r["person_switch_per_min"] == 0.0
    assert r["n_pose"] == T and r["multi_person_pct"] == 0.0


def test_qc_detects_spikes():
    kps, valid, boxes, n = make_trial()
    idx = np.arange(50, T - 50, 100)[:18]
    kps[idx, 15, 0] += 150
    r = run(kps, valid, boxes, n)
    assert r["spike_pct"] > 0
    ev = event_frames(kps, valid, boxes, LB6)
    assert set(idx.tolist()) <= set(ev["spike"].tolist())
    assert r["lr_swap_per_min"] == 0.0


def test_qc_detects_swaps():
    kps, valid, boxes, n = make_trial()
    for t0 in range(100, 1100, 100):
        for l, rj in ((13, 14), (15, 16)):
            kps[t0, [l, rj]] = kps[t0, [rj, l]]
    r = run(kps, valid, boxes, n)
    assert r["lr_swap_per_min"] >= 10
    assert "lr_swap" in r["flags"].split(";")


def test_qc_detects_person_switch():
    kps, valid, boxes, n = make_trial()
    for t0 in (200, 500, 800, 1100, 1400):
        boxes[t0] = [0, 0, 100, 100]
    r = run(kps, valid, boxes, n)
    assert r["person_switch_per_min"] > 1
    assert "person_switch" in r["flags"].split(";")


def test_qc_flags_frame_mismatch():
    r = run(*make_trial(), n_label30=T + 3)
    assert "frame_mismatch" in r["flags"].split(";")
    assert r["n_label30"] == T + 3


def test_frame_mismatch_tolerates_one_label_row():
    # 5 real trials have exactly one label row fewer than video frames; Rhama's min() handles it
    r = run(*make_trial(), n_label30=T - 1)
    assert "frame_mismatch" not in r["flags"].split(";")


def test_video_mismatch_flag():
    r = run(*make_trial(), n_video=T + 50)
    assert "video_mismatch" in r["flags"].split(";")
    r = run(*make_trial(), n_video=T + 1)
    assert "video_mismatch" not in r["flags"].split(";")


def test_feet_info_columns_not_flagged():
    kps, valid, boxes, n = make_trial()
    kps[:, 17:23, 2] = 0.05
    feet = {"L_BigToe": 17, "L_SmallToe": 18, "L_Heel": 19, "R_BigToe": 20, "R_SmallToe": 21, "R_Heel": 22}
    r = qc_trial(kps, valid, boxes, n, FPS, LB6, n_label30=T, n_video=T, info_joints=feet)
    for name in feet:
        assert r[f"lowconf_pct_{name}"] == 100.0
    assert r["flags"] == ""


def test_qc_trial_all_invalid():
    kps, valid, boxes, n = make_trial()
    kps[:] = 0
    valid[:] = False
    boxes[:] = 0
    n[:] = 0
    r = run(kps, valid, boxes, n)
    assert r["valid_pct"] == 0.0
    assert "valid" in r["flags"].split(";")
    for k in ("spike_pct", "lr_swap_per_min", "person_switch_per_min"):
        assert r[k] == 0.0


def test_qc_lowconf():
    kps, valid, boxes, n = make_trial()
    kps[: T // 5, 15, 2] = 0.1
    r = run(kps, valid, boxes, n)
    assert abs(r["lowconf_pct_L_Ankle"] - 20.0) < 0.1
    assert r["lowconf_pct_R_Ankle"] == 0.0
    assert "lowconf" in r["flags"].split(";")


def test_qc_without_n_persons():
    kps, valid, boxes, _ = make_trial()
    r = run(kps, valid, boxes, None)
    assert np.isnan(r["multi_person_pct"])
