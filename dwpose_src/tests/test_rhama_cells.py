import difflib
from pathlib import Path

import numpy as np

from dwpose_src.rhama_cells import (
    REPLACEMENTS,
    RHAMA_CELL_IDS,
    load_rhama_cells,
    rhama_constants_block,
    transform_rhama_source,
)

REPO = Path(__file__).resolve().parents[2]
RHAMA_NB = REPO / "yolo_notebooks" / "01_extract_yolo26_patientlock.ipynb"


def _cells():
    return load_rhama_cells(RHAMA_NB)


def test_cell_ids_match_titles():
    cells = _cells()
    assert "trial_records" in cells[RHAMA_CELL_IDS["manifest"]]
    assert "bbox_npz_path" in cells[RHAMA_CELL_IDS["manifest"]]  # PatientLock version of the team notebook
    assert "TAHAP 2: FUNGSI PREPROCESSING" in cells[RHAMA_CELL_IDS["preprocess_fns"]]
    assert "TAHAP 3: SIMPAN SCENARIO" in cells[RHAMA_CELL_IDS["scenarios"]]
    assert "TAHAP 4: SMOKE TEST" in cells[RHAMA_CELL_IDS["smoke"]]


def test_transform_counts():
    cells = _cells()
    src6, n6 = transform_rhama_source(cells[RHAMA_CELL_IDS["scenarios"]])
    src7, n7 = transform_rhama_source(cells[RHAMA_CELL_IDS["smoke"]])
    assert n6 == {"yolo26n_lb6_": 3, "mobilefog_rhama": 0, "default unggulan Rhama,": 1}
    assert n7 == {"yolo26n_lb6_": 0, "mobilefog_rhama": 1, "default unggulan Rhama,": 0}
    assert "mobilefog_dwpose_patientlock" in src7
    assert "yolo26n" not in src6 and "Rhama" not in src6
    assert "yolo26n" not in src7 and "rhama" not in src7.lower()
    assert "dwpose_m_lb6_" in src6
    assert "mobilefog_dwpose" in src7


def test_transform_is_minimal():
    orig = _cells()[RHAMA_CELL_IDS["scenarios"]]
    new, _ = transform_rhama_source(orig)
    removed = [l[2:] for l in difflib.ndiff(orig.splitlines(), new.splitlines()) if l.startswith("- ")]
    added = [l[2:] for l in difflib.ndiff(orig.splitlines(), new.splitlines()) if l.startswith("+ ")]
    assert len(removed) == len(added) == 4
    for r, a in zip(removed, added):
        expected = r
        for old, new in REPLACEMENTS.items():
            expected = expected.replace(old, new)
        assert expected == a and expected != r
    # the only non-path change is inside a comment line
    comment_changes = [r for r in removed if "yolo26n_lb6_" not in r]
    assert len(comment_changes) == 1 and comment_changes[0].lstrip().startswith("#")


def test_constants_values():
    cell2 = _cells()[2]
    ns: dict = {}
    exec(rhama_constants_block(cell2), ns)
    assert ns["LB6_INDICES"] == [11, 12, 13, 14, 15, 16]
    assert ns["LB6_NAMES"] == ["L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"]
    assert ns["CONF_VALID_THRES"] == 0.20
    assert ns["HAMPEL_WINDOW"] == 7
    assert ns["HAMPEL_SIGMA"] == 3.0
    assert ns["BUTTER_CUTOFF_HZ"] == 9.0
    assert ns["WINDOW_SEC"] == 2.0
    assert ns["STRIDE_SEC"] == 1.0
    assert ns["RANDOM_SEED"] == 42
    assert ns["DET_CONF"] == 0.20 and ns["CROP_PAD_RATIO"] == 0.15
    assert ns["MASK_SUPERVISORS"] is True and ns["MIN_CROP_IOU"] == 0.20
    # nothing from the YOLO-specific config leaks in
    assert "YOLO_WEIGHTS" not in ns and "DATASET_ROOT" not in ns


def _synthetic_133(T=300, fs=30.0):
    t = np.arange(T) / fs
    kps = np.zeros((T, 133, 3), np.float32)
    kps[..., 2] = 0.9
    base = {11: (600, 300), 12: (680, 300), 13: (590, 450), 14: (690, 450), 15: (585, 600), 16: (695, 600)}
    for j, (x, y) in base.items():
        phase = 0.0 if j % 2 else np.pi
        kps[:, j, 0] = x + 15 * np.sin(2 * np.pi * 1.0 * t + phase)
        kps[:, j, 1] = y + 5 * np.cos(2 * np.pi * 1.0 * t + phase)
    return kps


def test_preprocess_runs_on_133kp():
    cells = _cells()
    ns: dict = {}
    exec("import numpy as np\nimport pandas as pd\n", ns)
    exec(rhama_constants_block(cells[2]), ns)
    exec(cells[RHAMA_CELL_IDS["preprocess_fns"]], ns)
    a, b = ns["build_scenario_a_and_b"](_synthetic_133(), fs=30.0)
    assert a.shape == (300, 6, 7)
    assert b.shape == (300, 6, 9)
    assert not np.isnan(a).any() and not np.isnan(b).any()
