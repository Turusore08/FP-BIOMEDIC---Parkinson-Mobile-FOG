import ast
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from dwpose_src import build_notebook as bn
from dwpose_src import extract_worker as ew
from dwpose_src.nb_helpers import (eta_hours, find_bbox_root, find_dataset_root, require_cuda_provider,
                                      restore_previous_outputs)
from dwpose_src.rhama_cells import RHAMA_CELL_IDS, load_rhama_cells, transform_rhama_source
from dwpose_src.tests.test_extract_worker import CFG, FakeModels, _make_bbox, _make_video

REPO = Path(__file__).resolve().parents[2]


# ---------- nb_helpers ----------
def test_require_cuda_provider_raises():
    with pytest.raises(RuntimeError, match="onnxruntime-gpu"):
        require_cuda_provider(["CPUExecutionProvider"])
    require_cuda_provider(["CUDAExecutionProvider", "CPUExecutionProvider"])


def test_find_dataset_root(tmp_path):
    root = tmp_path / "a" / "b" / "Train Val Test"
    (root / "train" / "videos").mkdir(parents=True)
    (root / "val" / "videos").mkdir(parents=True)
    assert find_dataset_root([tmp_path / "missing", tmp_path]) == root
    with pytest.raises(FileNotFoundError):
        find_dataset_root([tmp_path / "missing"])


def test_find_bbox_root(tmp_path):
    root = tmp_path / "input" / "mobile-fog-patient-bboxes"
    for sp in ("train", "val", "test"):
        (root / "patient_bboxes_npz" / sp).mkdir(parents=True)
    assert find_bbox_root([tmp_path / "missing", tmp_path / "input"]) == root
    with pytest.raises(FileNotFoundError, match="mobile-fog-patient-bboxes"):
        find_bbox_root([tmp_path / "missing"])


def test_restore_previous_outputs(tmp_path):
    src = tmp_path / "input" / "x" / "01_extracted_poses_raw" / "dwpose_m" / "train"
    src.mkdir(parents=True)
    for t in ("PDFE02_1", "PDFE03_1"):
        np.savez(src / f"{t}.npz", a=np.arange(2))
    raw = tmp_path / "raw"
    (raw / "train").mkdir(parents=True)
    np.savez(raw / "train" / "PDFE02_1.npz", a=np.arange(5))
    assert restore_previous_outputs(tmp_path / "input", raw) == 1
    assert (raw / "train" / "PDFE03_1.npz").exists()
    assert len(np.load(raw / "train" / "PDFE02_1.npz")["a"]) == 5  # existing file untouched
    assert restore_previous_outputs(tmp_path / "nope", raw) == 0


def test_eta_hours():
    assert abs(eta_hours(50.0, 256000, 2) - 1.7778) < 1e-3


# ---------- worker provider guard ----------
def test_check_providers_rejects_cpu_fallback():
    with pytest.raises(RuntimeError, match="CPU"):
        ew.check_providers(["CPUExecutionProvider"], "cuda")
    ew.check_providers(["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda")
    ew.check_providers(["CPUExecutionProvider"], "cpu")


def test_build_models_exposes_providers(monkeypatch):
    class FakeTool:
        def __init__(self, *a, **k):
            self.session = types.SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

        def __call__(self, *a, **k):
            return np.zeros((0, 4), np.float32)

    monkeypatch.setitem(sys.modules, "rtmlib", types.SimpleNamespace(YOLOX=FakeTool, RTMPose=FakeTool))
    det_fn, pose_fn = ew.build_models("d", "p", "cuda")
    assert det_fn.providers == ["CPUExecutionProvider"]
    assert pose_fn.providers == ["CPUExecutionProvider"]


# ---------- generated notebook ----------
@pytest.fixture(scope="module")
def nb(tmp_path_factory):
    out = tmp_path_factory.mktemp("nb") / "nb.ipynb"
    bn.build(out)
    return json.loads(out.read_text(encoding="utf-8"))


def _code(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_structure(nb):
    assert nb["nbformat"] == 4 and nb["metadata"]["kernelspec"]["name"] == "python3"
    code = _code(nb)
    for src in code:
        stripped = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        ast.parse(stripped)
    joined = "\n".join(code)
    assert "yolo26n" not in joined and "Wholebody(" not in joined and "mode=" not in joined
    assert bn.DET_URL in joined and bn.POSE_URL in joined
    assert "--no-deps rtmlib==0.0.16" in joined and "onnxruntime-gpu" in joined
    cells = load_rhama_cells(REPO / "yolo_notebooks" / "01_extract_yolo26_patientlock.ipynb")
    for key in ("manifest", "preprocess_fns", "scenarios", "smoke"):
        assert transform_rhama_source(cells[RHAMA_CELL_IDS[key]])[0] in code
    assert "BBOX_ROOT = find_bbox_root" in joined
    assert '"bbox_npz_path": r.bbox_npz_path' in joined
    for k in ("crop_pad_ratio", "mask_supervisors", "min_crop_iou", "det_conf"):
        assert f'"{k}"' in joined, k
    assert "mobilefog_dwpose_patientlock" in joined
    assert "select_main_person" not in joined and "body17" not in joined


NAMES = re.compile(r"(?<![a-z])(rhama|sultan|faiz|panji|arya|hamdi|jailani|daulay|hizbullah)(?![a-z])", re.I)


def test_notebook_has_no_personal_names(nb):
    hits = [ln.strip() for c in nb["cells"] for ln in "".join(c["source"]).splitlines() if NAMES.search(ln)]
    assert hits == []
    assert not NAMES.search(bn.OUT_NB.name)


def test_onnxruntime_gpu_pinned(nb):
    assert "onnxruntime-gpu==1.32.0" in "\n".join(_code(nb))


def test_inlined_sources_are_not_altered(tmp_path, monkeypatch):
    monkeypatch.setattr(bn, "worker_source", lambda: "s = '{{x}}'\n")
    monkeypatch.setattr(bn, "helpers_source", lambda: "h = '{{y}}'\n")
    out = tmp_path / "nb.ipynb"
    bn.build(out)
    joined = "\n".join(_code(json.loads(out.read_text(encoding="utf-8"))))
    assert "s = '{{x}}'" in joined
    assert "h = '{{y}}'" in joined
    assert bn.qc_source().strip() in joined


def test_inlined_worker_runs_standalone(tmp_path):
    src = bn.worker_source()
    assert "inline-strip" not in src and "dwpose_src" not in src
    ns = {"__name__": "dwpose_worker"}
    exec(compile(src, "worker", "exec"), ns)
    vp = _make_video(tmp_path / "v.mp4")
    bp = _make_bbox(tmp_path / "b.npz")
    m = FakeModels()
    out = ns["extract_video"](str(vp), str(bp), m.det_fn, m.pose_fn, **CFG)
    assert out["kps"].shape == (12, 133, 3) and out["valid"].all()
    assert str(out["selector"]) == "patientlock-2pass"


def test_inlined_qc_runs_standalone(tmp_path):
    src = bn.qc_source()
    assert "inline-strip" not in src and "dwpose_src" not in src
    ns = {"__name__": "qc"}
    exec(compile(src, "qc", "exec"), ns)
    for name in ("qc_trial", "event_frames", "plot_signals", "render_clip", "clip_windows",
                 "pick_sample_trials", "write_review_template"):
        assert callable(ns[name])


def test_config_and_manifest_cells_run_locally(nb, tmp_path, monkeypatch):
    """Cells 1(helpers part)+2+3 locally: finds dataset/, audits 71 trials, writes manifest."""
    monkeypatch.chdir(REPO)
    monkeypatch.setenv("MOBILEFOG_OUTPUT_ROOT", str(tmp_path / "out"))
    bb = tmp_path / "bboxes"
    for vp in (REPO / "dataset").glob("*/videos/*.mp4"):
        d = bb / "patient_bboxes_npz" / vp.parent.parent.name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{vp.stem}.npz").write_bytes(b"")
    monkeypatch.setenv("MOBILEFOG_BBOX_ROOT", str(bb))
    ns = {"__name__": "__main__"}
    exec("from pathlib import Path\nimport os, json\nimport numpy as np\nimport pandas as pd\n", ns)
    exec(bn.helpers_source(), ns)
    cfg_cell = next(s for s in _code(nb) if "DATASET_ROOT = find_dataset_root" in s)
    exec(cfg_cell, ns)
    manifest_cell = next(s for s in _code(nb) if "trial_records" in s)
    exec(manifest_cell, ns)
    assert len(ns["df_manifest"]) == 71
    assert (tmp_path / "out" / "manifest_trials_local.csv").exists()
    assert ns["RAW_POSE_DIR"] == tmp_path / "out" / "01_extracted_poses_raw" / "dwpose_m"
    assert ns["BBOX_ROOT"] == bb
    assert all(Path(p).exists() for p in ns["df_manifest"]["bbox_npz_path"])
